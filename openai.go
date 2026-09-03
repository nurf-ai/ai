package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"errors"

	"github.com/567-labs/instructor-go/pkg/instructor"
	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

type OpenAIProvider struct {
	instructor *instructor.InstructorOpenAI
	raw        *openai.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
	// toolsNoReasoning is set after the API refuses function tools together
	// with reasoning_effort on /v1/chat/completions (some gpt-5.x variants):
	// tool calls then go out with reasoning_effort "none" from the start.
	toolsNoReasoning bool
}

func (p *OpenAIProvider) WithModeration(m ModerationProvider) *OpenAIProvider {
	p.moderation = m
	return p
}

func (p *OpenAIProvider) WithMeter(hook MeterHook) *OpenAIProvider {
	p.meter = hook
	return p
}

func (p *OpenAIProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *OpenAIProvider) SetModeration(m ModerationProvider) { p.moderation = m }

func NewOpenAIProvider(apiKey, model string) *OpenAIProvider {
	return newOpenAIProvider(openai.NewClient(apiKey), model)
}

// NewOpenAIProviderWithBaseURL targets an OpenAI-compatible endpoint (proxy,
// gateway, test server) instead of api.openai.com.
func NewOpenAIProviderWithBaseURL(apiKey, model, baseURL string) *OpenAIProvider {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	return newOpenAIProvider(openai.NewClientWithConfig(cfg), model)
}

func newOpenAIProvider(raw *openai.Client, model string) *OpenAIProvider {
	client := instructor.FromOpenAI(
		raw,
		instructor.WithMode(instructor.ModeJSON),
		instructor.WithMaxRetries(3),
	)
	return &OpenAIProvider{
		instructor: client,
		raw:        raw,
		model:      model,
	}
}

func (p *OpenAIProvider) Name() string              { return "OpenAI" }
func (p *OpenAIProvider) Model() string             { return p.model }
func (p *OpenAIProvider) RawClient() *openai.Client { return p.raw }

func isReasoningModel(model string) bool {
	m := strings.ToLower(model)
	return strings.HasPrefix(m, "o1") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4") || strings.Contains(m, "gpt-5")
}

// MaxInputTokens returns the advertised input context window for p.model.
func (p *OpenAIProvider) MaxInputTokens() (int64, error) {
	return MaxInputTokensLLM("openai", p.model)
}

func (p *OpenAIProvider) emitUsage(ctx context.Context, usage openai.Usage, sysPrompt, userPrompt string) {
	if p.meter == nil {
		return
	}
	in := usage.PromptTokens
	out := usage.CompletionTokens
	// OpenAI auto-caches prompts ≥1024 tokens; the cached portion is reported
	// in PromptTokensDetails.CachedTokens (nil-safe — older API versions may
	// omit the field). No separate cache_creation premium on OpenAI.
	cacheRead := 0
	if usage.PromptTokensDetails != nil {
		cacheRead = usage.PromptTokensDetails.CachedTokens
	}
	ev := UsageEvent{
		CallerID:             MeterCallerIDFromCtx(ctx),
		Provider:             "openai",
		Model:                p.model,
		Operation:            MeterOperationFromCtx(ctx),
		InputTokens:          in,
		OutputTokens:         out,
		TotalTokens:          in + out,
		CacheReadInputTokens: cacheRead,
		EstimatedCostUSD:     EstimateCostFull(p.model, in, out, 0, cacheRead),
		SystemPrompt:         TruncatePromptForDebug(sysPrompt),
		UserPrompt:           TruncatePromptForDebug(userPrompt),
		DebugSpanID:          DebugSpanIDFromCtx(ctx),
	}
	attachBlocks(ctx, &ev)
	ev.Metadata = mergeMeterMetadata(ctx, ev.Metadata)
	p.meter(ev)
}

func (p *OpenAIProvider) CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error {
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output",
		zap.String("provider", "openai"),
		zap.String("model", p.model),
		zap.String("userPrompt", userPrompt),
		zap.String("outputType", fmt.Sprintf("%T", structuredOutput)),
		zap.String("sysPrompt", sysPrompt),
	)
	msgs := []openai.ChatCompletionMessage{}
	if sysPrompt != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: sysPrompt})
	}
	msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: userPrompt})
	_, err := p.instructor.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    p.model,
			Messages: msgs,
		},
		structuredOutput,
	)
	if err != nil {
		return fmt.Errorf("openai structured output: %w", err)
	}
	return nil
}

// CreateStructuredOutputBreakpointed satisfies router.CachedStructuredLLM.
// OpenAI auto-prefix-caches any stable prefix ≥1024 tokens, so the
// breakpointing surf is implemented by concatenating sysPrompt and
// stableMid into the single system message. No explicit markers needed —
// the byte-stable prefix is the cache key.
//
// dynamicTail rides as the user message (uncached).
func (p *OpenAIProvider) CreateStructuredOutputBreakpointed(
	ctx context.Context,
	sysPrompt, stableMid, dynamicTail string,
	structuredOutput any,
) error {
	combinedSys := sysPrompt
	if stableMid != "" {
		if combinedSys != "" {
			combinedSys += "\n\n"
		}
		combinedSys += stableMid
	}
	return p.CreateStructuredOutput(ctx, dynamicTail, combinedSys, structuredOutput)
}

func (p *OpenAIProvider) CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	return p.CreateStructuredOutputFromParts(ctx, []Part{TextPart{Text: userPrompt}}, sysPrompt, schema)
}

// CreateStructuredOutputFromParts is CreateStructuredOutputFromSchema with a
// multimodal user turn (text + base64 images). Honours WithMaxTokens
// (default 4096) and forces the structured_output tool so the model cannot
// answer in prose.
func (p *OpenAIProvider) CreateStructuredOutputFromParts(ctx context.Context, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	userText, _ := PartsText(parts)
	if err := checkModeration(ctx, p.moderation, userText); err != nil {
		return nil, err
	}
	logger.Log(traceLevel, "structured output from schema", zap.String("provider", "openai"), zap.String("model", p.model), zap.Int("parts", len(parts)))

	var schemaObj map[string]any
	if err := json.Unmarshal(schema, &schemaObj); err != nil {
		return nil, fmt.Errorf("invalid schema JSON: %w", err)
	}

	maxOut := MaxTokensFromCtx(ctx, 4096)
	msgs := []openai.ChatCompletionMessage{}
	if sysPrompt != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: sysPrompt})
	}
	msgs = append(msgs, openaiUserMessageFromParts(parts))
	req := openai.ChatCompletionRequest{
		Model:    p.model,
		Messages: msgs,
		Tools: []openai.Tool{
			{
				Type: openai.ToolTypeFunction,
				Function: &openai.FunctionDefinition{
					Name:        "structured_output",
					Description: "Generate structured output matching the schema",
					Parameters:  schemaObj,
				},
			},
		},
		ToolChoice: openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: "structured_output"},
		},
		MaxCompletionTokens: maxOut,
	}
	if isReasoningModel(p.model) {
		req.ReasoningEffort = "none"
	}

	completion, err := p.raw.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai function call failed: %w", err)
	}
	p.emitUsage(ctx, completion.Usage, sysPrompt, userText)

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("openai: no choices returned")
	}

	choice := completion.Choices[0]
	if choice.FinishReason == openai.FinishReasonLength {
		return nil, fmt.Errorf("structured output truncated at max_completion_tokens=%d (finish_reason=length): shrink the schema/output or raise the budget with ai.WithMaxTokens", maxOut)
	}
	if choice.FinishReason == openai.FinishReasonContentFilter {
		// also fires when the model starts reproducing well-known text verbatim (regurgitation guard)
		return nil, fmt.Errorf("structured output stopped by the provider content filter after %d tokens (finish_reason=content_filter): ask for paraphrases instead of verbatim quotes", completion.Usage.CompletionTokens)
	}
	for _, tc := range choice.Message.ToolCalls {
		if tc.Function.Name == "structured_output" {
			var result map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &result); err != nil {
				return nil, fmt.Errorf("unmarshal function args (finish_reason=%s, args_len=%d, completion_tokens=%d): %w",
					choice.FinishReason, len(tc.Function.Arguments), completion.Usage.CompletionTokens, err)
			}
			return result, nil
		}
	}

	return nil, fmt.Errorf("no structured output in response (finish_reason=%s, tool_calls=%d, content=%q)",
		choice.FinishReason, len(choice.Message.ToolCalls), truncate(strings.TrimSpace(choice.Message.Content), 200))
}

// openaiUserMessageFromParts builds one user message from a multimodal turn.
// Text-only turns keep the plain Content form — Content and MultiContent are
// mutually exclusive on the wire, and some OpenAI-compatible hosts reject an
// all-text MultiContent array.
func openaiUserMessageFromParts(parts []Part) openai.ChatCompletionMessage {
	text, hasImage := PartsText(parts)
	if !hasImage {
		return openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: text}
	}
	mc := make([]openai.ChatMessagePart, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ImagePart:
			mc = append(mc, openai.ChatMessagePart{
				Type:     openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{URL: "data:" + v.MediaType + ";base64," + v.Data},
			})
		case TextPart:
			mc = append(mc, openai.ChatMessagePart{Type: openai.ChatMessagePartTypeText, Text: v.Text})
		}
	}
	return openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, MultiContent: mc}
}

func (p *OpenAIProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
	for _, m := range messages {
		if m.Role == RoleUser {
			if err := checkModeration(ctx, p.moderation, m.Content); err != nil {
				return nil, err
			}
		}
	}
	toolNames := make([]string, len(tools))
	for i, t := range tools {
		toolNames[i] = t.Name
	}
	logger.Log(traceLevel, "chat", zap.String("provider", "openai"), zap.String("model", p.model), zap.Int("messages", len(messages)), zap.Any("tools", toolNames))
	var apiMessages []openai.ChatCompletionMessage

	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			apiMessages = append(apiMessages, openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleSystem,
				Content: m.Content,
			})

		case RoleUser:
			if len(m.Parts) > 0 {
				var parts []openai.ChatMessagePart
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						parts = append(parts, openai.ChatMessagePart{
							Type: openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{
								URL: "data:" + v.MediaType + ";base64," + v.Data,
							},
						})
					case TextPart:
						parts = append(parts, openai.ChatMessagePart{
							Type: openai.ChatMessagePartTypeText,
							Text: v.Text,
						})
					}
				}
				apiMessages = append(apiMessages, openai.ChatCompletionMessage{
					Role:         openai.ChatMessageRoleUser,
					MultiContent: parts,
				})
			} else {
				apiMessages = append(apiMessages, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: m.Content,
				})
			}

		case RoleAssistant:
			msg := openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: m.Content,
			}
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Arguments),
					},
				})
			}
			apiMessages = append(apiMessages, msg)

		case RoleTool:
			apiMessages = append(apiMessages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    m.Content,
				ToolCallID: m.ToolCallID,
			})
		}
	}

	var apiTools []openai.Tool
	for _, t := range tools {
		apiTools = append(apiTools, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	req := openai.ChatCompletionRequest{
		Model:               p.model,
		Messages:            apiMessages,
		MaxCompletionTokens: 4096,
	}
	if effort := ReasoningEffortFromCtx(ctx); effort != "" {
		req.ReasoningEffort = effort
	}
	if len(apiTools) > 0 {
		req.Tools = apiTools
		if p.toolsNoReasoning {
			req.ReasoningEffort = "none"
		}
	}

	completion, err := p.raw.CreateChatCompletion(ctx, req)
	if err != nil && toolsRejectReasoning(err, len(apiTools) > 0, req.ReasoningEffort) {
		// the model wants reasoning off for function tools on this endpoint: once, then sticky
		p.toolsNoReasoning = true
		req.ReasoningEffort = "none"
		completion, err = p.raw.CreateChatCompletion(ctx, req)
	}
	if err != nil {
		return nil, fmt.Errorf("openai chat failed: %w", err)
	}
	// Reconstruct sys/user prompts from messages for debug capture.
	var sysText, userText strings.Builder
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			if sysText.Len() > 0 {
				sysText.WriteString("\n\n")
			}
			sysText.WriteString(m.Content)
		case RoleUser:
			if userText.Len() > 0 {
				userText.WriteString("\n\n")
			}
			if m.Content != "" {
				userText.WriteString(m.Content)
			}
			for _, pt := range m.Parts {
				if v, ok := pt.(TextPart); ok {
					userText.WriteString(v.Text)
				}
			}
		}
	}
	p.emitUsage(ctx, completion.Usage, sysText.String(), userText.String())

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("openai chat: no choices returned")
	}

	choice := completion.Choices[0]
	resp := &Response{Content: choice.Message.Content}

	for _, tc := range choice.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}

	return resp, nil
}

const defaultEmbeddingModel = "text-embedding-3-small"
const defaultEmbeddingDims = 1536

func (p *OpenAIProvider) EmbedText(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := p.raw.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: texts,
		Model: openai.EmbeddingModel(defaultEmbeddingModel),
	})
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

func (p *OpenAIProvider) EmbedDimensions() int { return defaultEmbeddingDims }

func (p *OpenAIProvider) ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error) {
	for _, m := range messages {
		if m.Role == RoleUser {
			if err := checkModeration(ctx, p.moderation, m.Content); err != nil {
				return nil, err
			}
		}
	}

	var apiMessages []openai.ChatCompletionMessage
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleSystem, Content: m.Content})
		case RoleUser:
			if len(m.Parts) > 0 {
				var parts []openai.ChatMessagePart
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						parts = append(parts, openai.ChatMessagePart{Type: openai.ChatMessagePartTypeImageURL, ImageURL: &openai.ChatMessageImageURL{URL: "data:" + v.MediaType + ";base64," + v.Data}})
					case TextPart:
						parts = append(parts, openai.ChatMessagePart{Type: openai.ChatMessagePartTypeText, Text: v.Text})
					}
				}
				apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, MultiContent: parts})
			} else {
				apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: m.Content})
			}
		case RoleAssistant:
			msg := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: m.Content}
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{ID: tc.ID, Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: tc.Name, Arguments: string(tc.Arguments)}})
			}
			apiMessages = append(apiMessages, msg)
		case RoleTool:
			apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, Content: m.Content, ToolCallID: m.ToolCallID})
		}
	}

	var apiTools []openai.Tool
	for _, t := range tools {
		apiTools = append(apiTools, openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: t.Name, Description: t.Description, Parameters: t.Parameters}})
	}

	req := openai.ChatCompletionRequest{Model: p.model, Messages: apiMessages, MaxCompletionTokens: 4096}
	if effort := ReasoningEffortFromCtx(ctx); effort != "" {
		req.ReasoningEffort = effort
	}
	if len(apiTools) > 0 {
		req.Tools = apiTools
		if p.toolsNoReasoning {
			req.ReasoningEffort = "none"
		}
	}

	resp, usage, err := openaiStreamLoop(ctx, p.raw, req, cb)
	if err != nil && toolsRejectReasoning(err, len(apiTools) > 0, req.ReasoningEffort) {
		p.toolsNoReasoning = true
		req.ReasoningEffort = "none"
		resp, usage, err = openaiStreamLoop(ctx, p.raw, req, cb)
	}
	if err != nil && !errors.Is(err, errStreamBreak) {
		return nil, fmt.Errorf("openai stream: %w", err)
	}

	sysText, userText := extractPrompts(messages)
	p.emitUsage(ctx, usage, sysText, userText)
	return resp, nil
}

// toolsRejectReasoning recognises the 400 some gpt-5.x variants return when a
// chat-completions request carries function tools while reasoning is on
// ("Function tools with reasoning_effort are not supported for <model> …
// set reasoning_effort to 'none'"). Nothing was sent to the model, so a retry
// with reasoning_effort "none" is safe.
func toolsRejectReasoning(err error, hasTools bool, effort string) bool {
	if err == nil || !hasTools || effort == "none" {
		return false
	}
	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) || apiErr.HTTPStatusCode != 400 {
		return false
	}
	msg := strings.ToLower(apiErr.Message)
	return strings.Contains(msg, "reasoning_effort") && strings.Contains(msg, "tool")
}
