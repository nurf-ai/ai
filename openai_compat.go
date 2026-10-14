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

// openAICompat is the shared core of providers that speak an
// OpenAI-compatible Chat Completions API under their own name and
// models.json group (Hugging Face, OpenRouter).
type openAICompat struct {
	instructor *instructor.InstructorOpenAI
	raw        *openai.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
	provider   string // models.json group + UsageEvent.Provider
	name       string // Name()
}

func newOpenAICompat(cfg openai.ClientConfig, model, provider, name string) openAICompat {
	raw := openai.NewClientWithConfig(cfg)
	client := instructor.FromOpenAI(
		raw,
		instructor.WithMode(instructor.ModeJSON),
		instructor.WithMaxRetries(3),
	)
	return openAICompat{
		instructor: client,
		raw:        raw,
		model:      model,
		provider:   provider,
		name:       name,
	}
}

func (p *openAICompat) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *openAICompat) SetModeration(m ModerationProvider) { p.moderation = m }

func (p *openAICompat) Name() string              { return p.name }
func (p *openAICompat) Model() string             { return p.model }
func (p *openAICompat) RawClient() *openai.Client { return p.raw }

// MaxInputTokens returns the advertised input context window for p.model.
func (p *openAICompat) MaxInputTokens() (int64, error) {
	return MaxInputTokensLLM(p.provider, p.model)
}

func (p *openAICompat) emitUsage(ctx context.Context, usage openai.Usage, sysPrompt, userPrompt string) {
	if p.meter == nil {
		return
	}
	in := usage.PromptTokens
	out := usage.CompletionTokens
	cost, reported := reportedCostFromCtx(ctx)
	if !reported {
		cost = EstimateCostFull(p.model, in, out, 0, 0)
	}
	ev := UsageEvent{
		CallerID:         MeterCallerIDFromCtx(ctx),
		Provider:         p.provider,
		Model:            p.model,
		Operation:        MeterOperationFromCtx(ctx),
		InputTokens:      in,
		OutputTokens:     out,
		TotalTokens:      in + out,
		EstimatedCostUSD: cost,
		SystemPrompt:     TruncatePromptForDebug(sysPrompt),
		UserPrompt:       TruncatePromptForDebug(userPrompt),
		DebugSpanID:      DebugSpanIDFromCtx(ctx),
	}
	attachBlocks(ctx, &ev)
	ev.Metadata = mergeMeterMetadata(ctx, ev.Metadata)
	p.meter(ev)
}

func (p *openAICompat) CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error {
	ctx = withReportedCost(ctx)
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output",
		zap.String("provider", p.provider),
		zap.String("model", p.model),
		zap.String("userPrompt", userPrompt),
		zap.String("outputType", fmt.Sprintf("%T", structuredOutput)),
		zap.String("sysPrompt", sysPrompt),
	)
	resp, err := p.instructor.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: p.model,
			Messages: []openai.ChatCompletionMessage{
				{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
				{Role: openai.ChatMessageRoleUser, Content: userPrompt},
			},
		},
		structuredOutput,
	)
	// instructor sums usage over its retries, and reports it on failure too
	if resp.Usage.TotalTokens > 0 || resp.Usage.PromptTokens > 0 {
		p.emitUsage(ctx, resp.Usage, sysPrompt, userPrompt)
	}
	if err != nil {
		return fmt.Errorf("%s structured output: %w", p.provider, err)
	}
	return nil
}

func (p *openAICompat) CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	return p.CreateStructuredOutputFromParts(ctx, []Part{TextPart{Text: userPrompt}}, sysPrompt, schema)
}

// CreateStructuredOutputFromParts is CreateStructuredOutputFromSchema with a
// multimodal user turn (text + base64 images).
func (p *openAICompat) CreateStructuredOutputFromParts(ctx context.Context, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	ctx = withReportedCost(ctx)
	userText, _ := PartsText(parts)
	if err := checkModeration(ctx, p.moderation, userText); err != nil {
		return nil, err
	}
	logger.Log(traceLevel, "structured output from schema", zap.String("provider", p.provider), zap.String("model", p.model), zap.Int("parts", len(parts)))

	var schemaObj map[string]any
	if err := json.Unmarshal(schema, &schemaObj); err != nil {
		return nil, fmt.Errorf("invalid schema JSON: %w", err)
	}

	maxOut := MaxTokensFromCtx(ctx, 4096)
	req := openai.ChatCompletionRequest{
		Model: p.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: sysPrompt},
			openaiUserMessageFromParts(parts),
		},
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

	completion, err := p.raw.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("%s function call failed: %w", p.provider, err)
	}
	p.emitUsage(ctx, completion.Usage, sysPrompt, userText)

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("%s: no choices returned", p.provider)
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

	// Fallback: some hosted models return structured JSON in content
	// instead of via tool calls.
	content := strings.TrimSpace(choice.Message.Content)
	if content != "" {
		if raw, ok := extractJSONObject(content); ok {
			var result map[string]any
			if err := json.Unmarshal(raw, &result); err == nil {
				return result, nil
			}
		}
	}

	return nil, fmt.Errorf("no structured output in response (finish_reason=%s, tool_calls=%d, content=%q)",
		choice.FinishReason, len(choice.Message.ToolCalls), truncate(content, 200))
}

func (p *openAICompat) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
	ctx = withReportedCost(ctx)
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
	logger.Log(traceLevel, "chat", zap.String("provider", p.provider), zap.String("model", p.model), zap.Int("messages", len(messages)), zap.Any("tools", toolNames))

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
	if len(apiTools) > 0 {
		req.Tools = apiTools
	}

	completion, err := p.raw.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("%s chat failed: %w", p.provider, err)
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
		return nil, fmt.Errorf("%s chat: no choices returned", p.provider)
	}

	choice := completion.Choices[0]
	resp := &Response{Content: choice.Message.Content, StopReason: openAIStopReason(string(choice.FinishReason))}

	for _, tc := range choice.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	resp.cutLastToolCall()

	return resp, nil
}

func (p *openAICompat) ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error) {
	ctx = withReportedCost(ctx)
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
	if len(apiTools) > 0 {
		req.Tools = apiTools
	}

	resp, usage, err := openaiStreamLoop(ctx, p.raw, req, cb)
	// a stream cut short is billed too (stream_partial.go)
	cut := err != nil
	sysText, userText := extractPrompts(messages)
	if uctx, u, ok := openAIStreamUsage(ctx, messages, resp, usage, cut); ok {
		p.emitUsage(uctx, u, sysText, userText)
	}
	if err != nil && !errors.Is(err, errStreamBreak) {
		return nil, fmt.Errorf("%s stream: %w", p.provider, err)
	}
	return resp, nil
}
