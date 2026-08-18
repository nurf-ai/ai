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

// OllamaProvider wraps the OpenAI-compatible API exposed by Ollama.
type OllamaProvider struct {
	instructor *instructor.InstructorOpenAI
	raw        *openai.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func (p *OllamaProvider) WithModeration(m ModerationProvider) *OllamaProvider {
	p.moderation = m
	return p
}

func (p *OllamaProvider) WithMeter(hook MeterHook) *OllamaProvider {
	p.meter = hook
	return p
}

func (p *OllamaProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *OllamaProvider) SetModeration(m ModerationProvider) { p.moderation = m }

// NewOllamaProvider creates a provider pointing at an Ollama instance.
// baseURL is the Ollama server, e.g. "http://ollama:11434/v1".
func NewOllamaProvider(baseURL, model string) *OllamaProvider {
	cfg := openai.DefaultConfig("")
	cfg.BaseURL = baseURL
	raw := openai.NewClientWithConfig(cfg)
	client := instructor.FromOpenAI(
		raw,
		instructor.WithMode(instructor.ModeJSON),
		instructor.WithMaxRetries(3),
	)
	return &OllamaProvider{
		instructor: client,
		raw:        raw,
		model:      model,
	}
}

func (p *OllamaProvider) Name() string  { return "Ollama" }
func (p *OllamaProvider) Model() string { return p.model }

func (p *OllamaProvider) emitUsage(ctx context.Context, usage openai.Usage, sysPrompt, userPrompt string) {
	if p.meter == nil {
		return
	}
	in := usage.PromptTokens
	out := usage.CompletionTokens
	ev := UsageEvent{
		CallerID:     MeterCallerIDFromCtx(ctx),
		Provider:     "ollama",
		Model:        p.model,
		Operation:    MeterOperationFromCtx(ctx),
		InputTokens:  in,
		OutputTokens: out,
		TotalTokens:  in + out,
		SystemPrompt: TruncatePromptForDebug(sysPrompt),
		UserPrompt:   TruncatePromptForDebug(userPrompt),
		DebugSpanID:  DebugSpanIDFromCtx(ctx),
	}
	attachBlocks(ctx, &ev)
	p.meter(ev)
}

func (p *OllamaProvider) CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error {
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output", zap.String("provider", "ollama"), zap.String("model", p.model))
	_, err := p.instructor.CreateChatCompletion(
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
	if err != nil {
		return fmt.Errorf("ollama structured output: %w", err)
	}
	return nil
}

func (p *OllamaProvider) CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return nil, err
	}
	logger.Log(traceLevel, "structured output from schema", zap.String("provider", "ollama"), zap.String("model", p.model))

	// Ollama models don't always honour tool calling, so prompt for JSON directly.
	jsonPrompt := fmt.Sprintf(
		"%s\n\nRespond with ONLY valid JSON matching this schema:\n%s",
		sysPrompt, string(schema),
	)

	resp, err := p.Chat(ctx, []Message{
		{Role: RoleSystem, Content: jsonPrompt},
		{Role: RoleUser, Content: userPrompt},
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("ollama schema output: %w", err)
	}

	raw, ok := extractJSONObject(resp.Content)
	if !ok {
		return nil, fmt.Errorf("ollama schema output: no JSON object found in: %s", resp.Content)
	}
	var result map[string]any
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("ollama schema output: invalid JSON: %w\nraw: %s", err, string(raw))
	}
	return result, nil
}

func (p *OllamaProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
	for _, m := range messages {
		if m.Role == RoleUser {
			if err := checkModeration(ctx, p.moderation, m.Content); err != nil {
				return nil, err
			}
		}
	}
	logger.Log(traceLevel, "chat", zap.String("provider", "ollama"), zap.String("model", p.model), zap.Int("messages", len(messages)))

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
				apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleUser, Content: m.Content})
			}
		case RoleAssistant:
			msg := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: m.Content}
			for _, tc := range m.ToolCalls {
				msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
					ID: tc.ID, Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{Name: tc.Name, Arguments: string(tc.Arguments)},
				})
			}
			apiMessages = append(apiMessages, msg)
		case RoleTool:
			apiMessages = append(apiMessages, openai.ChatCompletionMessage{Role: openai.ChatMessageRoleTool, Content: m.Content, ToolCallID: m.ToolCallID})
		}
	}

	var apiTools []openai.Tool
	for _, t := range tools {
		apiTools = append(apiTools, openai.Tool{
			Type:     openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{Name: t.Name, Description: t.Description, Parameters: t.Parameters},
		})
	}

	req := openai.ChatCompletionRequest{Model: p.model, Messages: apiMessages}
	if len(apiTools) > 0 {
		req.Tools = apiTools
	}

	completion, err := p.raw.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("ollama chat failed: %w", err)
	}
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
			userText.WriteString(m.Content)
		}
	}
	p.emitUsage(ctx, completion.Usage, sysText.String(), userText.String())

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("ollama chat: no choices returned")
	}

	choice := completion.Choices[0]
	resp := &Response{Content: stripThinkTags(choice.Message.Content)}
	for _, tc := range choice.Message.ToolCalls {
		resp.ToolCalls = append(resp.ToolCalls, ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	return resp, nil
}

func (p *OllamaProvider) ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error) {
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

	req := openai.ChatCompletionRequest{Model: p.model, Messages: apiMessages}
	if len(apiTools) > 0 {
		req.Tools = apiTools
	}

	resp, usage, err := openaiStreamLoop(ctx, p.raw, req, cb)
	if err != nil && !errors.Is(err, errStreamBreak) {
		return nil, fmt.Errorf("ollama stream: %w", err)
	}

	if resp != nil {
		resp.Content = stripThinkTags(resp.Content)
	}
	sysText, userText := extractPrompts(messages)
	p.emitUsage(ctx, usage, sysText, userText)
	return resp, nil
}
