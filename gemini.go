package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/invopop/jsonschema"
	"go.uber.org/zap"
	"google.golang.org/genai"
)

type GeminiProvider struct {
	client     *genai.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func (p *GeminiProvider) WithModeration(m ModerationProvider) *GeminiProvider {
	p.moderation = m
	return p
}

func (p *GeminiProvider) WithMeter(hook MeterHook) *GeminiProvider {
	p.meter = hook
	return p
}

func (p *GeminiProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *GeminiProvider) SetModeration(m ModerationProvider) { p.moderation = m }

func NewGeminiProvider(ctx context.Context, apiKey, model string) (*GeminiProvider, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	return &GeminiProvider{client: client, model: model}, nil
}

func (p *GeminiProvider) Name() string  { return "Gemini" }
func (p *GeminiProvider) Model() string { return p.model }

func (p *GeminiProvider) MaxInputTokens() (int64, error) {
	return MaxInputTokensLLM("gemini", p.model)
}

func (p *GeminiProvider) CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error {
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output",
		zap.String("provider", "gemini"),
		zap.String("model", p.model),
		zap.String("userPrompt", userPrompt),
		zap.String("outputType", fmt.Sprintf("%T", structuredOutput)),
		zap.String("sysPrompt", sysPrompt),
	)

	schema, err := structToGeminiSchema(structuredOutput)
	if err != nil {
		return fmt.Errorf("gemini schema reflect: %w", err)
	}

	config := &genai.GenerateContentConfig{
		ResponseMIMEType: "application/json",
		ResponseSchema:   schema,
		MaxOutputTokens:  int32(MaxTokensFromCtx(ctx, 4096)),
	}
	if sysPrompt != "" {
		config.SystemInstruction = genai.NewContentFromText(sysPrompt, genai.RoleUser)
	}

	contents := []*genai.Content{
		genai.NewContentFromText(userPrompt, genai.RoleUser),
	}
	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents, config)
	if err != nil {
		return fmt.Errorf("gemini structured output: %w", err)
	}
	p.emitUsage(ctx, resp, sysPrompt, userPrompt)

	text := extractGeminiText(resp)
	if text == "" {
		return fmt.Errorf("gemini: no text in response")
	}
	if err := json.Unmarshal([]byte(text), structuredOutput); err != nil {
		return fmt.Errorf("gemini unmarshal output: %w", err)
	}
	return nil
}

func (p *GeminiProvider) CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	return p.CreateStructuredOutputFromParts(ctx, []Part{TextPart{Text: userPrompt}}, sysPrompt, schema)
}

// CreateStructuredOutputFromParts is CreateStructuredOutputFromSchema with a
// multimodal user turn (text + base64 images).
func (p *GeminiProvider) CreateStructuredOutputFromParts(ctx context.Context, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	userText, _ := PartsText(parts)
	if err := checkModeration(ctx, p.moderation, userText); err != nil {
		return nil, err
	}
	logger.Log(traceLevel, "structured output from schema", zap.String("provider", "gemini"), zap.String("model", p.model), zap.Int("parts", len(parts)))

	config := &genai.GenerateContentConfig{
		ResponseMIMEType:   "application/json",
		ResponseJsonSchema: json.RawMessage(schema),
		MaxOutputTokens:    int32(MaxTokensFromCtx(ctx, 4096)),
	}
	if sysPrompt != "" {
		config.SystemInstruction = genai.NewContentFromText(sysPrompt, genai.RoleUser)
	}

	contents := []*genai.Content{{Parts: geminiPartsFromParts(parts), Role: "user"}}
	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("gemini structured output from schema: %w", err)
	}
	p.emitUsage(ctx, resp, sysPrompt, userText)

	text := extractGeminiText(resp)
	if text == "" {
		return nil, fmt.Errorf("gemini: no text in response")
	}
	var result map[string]any
	if err := json.Unmarshal([]byte(text), &result); err != nil {
		return nil, fmt.Errorf("gemini unmarshal schema output: %w", err)
	}
	return result, nil
}

// geminiPartsFromParts converts a multimodal turn into genai parts. ImagePart
// carries base64 text while genai.Blob wants raw bytes (the SDK encodes on the
// wire), so the image is decoded here.
func geminiPartsFromParts(parts []Part) []*genai.Part {
	out := make([]*genai.Part, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ImagePart:
			out = append(out, &genai.Part{InlineData: &genai.Blob{MIMEType: v.MediaType, Data: imagePartBytes(v)}})
		case TextPart:
			out = append(out, genai.NewPartFromText(v.Text))
		}
	}
	if len(out) == 0 {
		out = append(out, genai.NewPartFromText(""))
	}
	return out
}

func (p *GeminiProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
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
	logger.Log(traceLevel, "chat", zap.String("provider", "gemini"), zap.String("model", p.model), zap.Int("messages", len(messages)), zap.Any("tools", toolNames))

	config := &genai.GenerateContentConfig{
		MaxOutputTokens: int32(MaxTokensFromCtx(ctx, 4096)),
	}

	var contents []*genai.Content
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			config.SystemInstruction = genai.NewContentFromText(m.Content, genai.RoleUser)

		case RoleUser:
			if len(m.Parts) > 0 {
				var parts []*genai.Part
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						parts = append(parts, &genai.Part{
							InlineData: &genai.Blob{
								MIMEType: v.MediaType,
								Data:     imagePartBytes(v),
							},
						})
					case TextPart:
						parts = append(parts, genai.NewPartFromText(v.Text))
					}
				}
				contents = append(contents, &genai.Content{Parts: parts, Role: "user"})
			} else {
				contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleUser))
			}

		case RoleAssistant:
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, genai.NewPartFromText(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				json.Unmarshal(tc.Arguments, &args) //nolint:errcheck
				parts = append(parts, genai.NewPartFromFunctionCall(tc.Name, args))
			}
			contents = append(contents, &genai.Content{Parts: parts, Role: "model"})

		case RoleTool:
			var respData map[string]any
			json.Unmarshal([]byte(m.Content), &respData) //nolint:errcheck
			if respData == nil {
				respData = map[string]any{"result": m.Content}
			}
			contents = append(contents, genai.NewContentFromFunctionResponse(m.ToolCallID, respData, genai.RoleUser))
		}
	}

	if len(tools) > 0 {
		var decls []*genai.FunctionDeclaration
		for _, t := range tools {
			decls = append(decls, &genai.FunctionDeclaration{
				Name:                t.Name,
				Description:         t.Description,
				ParametersJsonSchema: t.Parameters,
			})
		}
		config.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}

	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("gemini chat: %w", err)
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
	p.emitUsage(ctx, resp, sysText.String(), userText.String())

	if resp == nil || len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("gemini chat: no candidates returned")
	}

	result := &Response{}
	candidate := resp.Candidates[0]
	if candidate.Content == nil {
		return result, nil
	}
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			if result.Content != "" {
				result.Content += "\n"
			}
			result.Content += part.Text
		}
		if part.FunctionCall != nil {
			args, _ := json.Marshal(part.FunctionCall.Args)
			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:        part.FunctionCall.ID,
				Name:      part.FunctionCall.Name,
				Arguments: args,
			})
		}
	}

	return result, nil
}

func (p *GeminiProvider) ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error) {
	for _, m := range messages {
		if m.Role == RoleUser {
			if err := checkModeration(ctx, p.moderation, m.Content); err != nil {
				return nil, err
			}
		}
	}

	config := &genai.GenerateContentConfig{
		MaxOutputTokens: int32(MaxTokensFromCtx(ctx, 4096)),
	}

	var contents []*genai.Content
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			config.SystemInstruction = genai.NewContentFromText(m.Content, genai.RoleUser)
		case RoleUser:
			if len(m.Parts) > 0 {
				var parts []*genai.Part
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: v.MediaType, Data: imagePartBytes(v)}})
					case TextPart:
						parts = append(parts, genai.NewPartFromText(v.Text))
					}
				}
				contents = append(contents, &genai.Content{Parts: parts, Role: "user"})
			} else {
				contents = append(contents, genai.NewContentFromText(m.Content, genai.RoleUser))
			}
		case RoleAssistant:
			var parts []*genai.Part
			if m.Content != "" {
				parts = append(parts, genai.NewPartFromText(m.Content))
			}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				json.Unmarshal(tc.Arguments, &args) //nolint:errcheck
				parts = append(parts, genai.NewPartFromFunctionCall(tc.Name, args))
			}
			contents = append(contents, &genai.Content{Parts: parts, Role: "model"})
		case RoleTool:
			var respData map[string]any
			json.Unmarshal([]byte(m.Content), &respData) //nolint:errcheck
			if respData == nil {
				respData = map[string]any{"result": m.Content}
			}
			contents = append(contents, genai.NewContentFromFunctionResponse(m.ToolCallID, respData, genai.RoleUser))
		}
	}

	if len(tools) > 0 {
		var decls []*genai.FunctionDeclaration
		for _, t := range tools {
			decls = append(decls, &genai.FunctionDeclaration{Name: t.Name, Description: t.Description, ParametersJsonSchema: t.Parameters})
		}
		config.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
	}

	var content strings.Builder
	var toolCalls []ToolCall
	var lastResp *genai.GenerateContentResponse

	for chunk, err := range p.client.Models.GenerateContentStream(ctx, p.model, contents, config) {
		if err != nil {
			return nil, fmt.Errorf("gemini stream: %w", err)
		}
		lastResp = chunk
		if chunk == nil || len(chunk.Candidates) == 0 {
			continue
		}
		candidate := chunk.Candidates[0]
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				content.WriteString(part.Text)
				if err := cb(StreamChunk{Text: part.Text}); err != nil {
					resp := &Response{Content: content.String()}
					if len(toolCalls) > 0 {
						resp.ToolCalls = toolCalls
					}
					return resp, err
				}
			}
			if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				toolCalls = append(toolCalls, ToolCall{ID: part.FunctionCall.ID, Name: part.FunctionCall.Name, Arguments: args})
			}
		}
	}

	sysText, userText := extractPrompts(messages)
	p.emitUsage(ctx, lastResp, sysText, userText)

	resp := &Response{Content: content.String()}
	if len(toolCalls) > 0 {
		resp.ToolCalls = toolCalls
	}
	return resp, nil
}

func (p *GeminiProvider) emitUsage(ctx context.Context, resp *genai.GenerateContentResponse, sysPrompt, userPrompt string) {
	if p.meter == nil || resp == nil || resp.UsageMetadata == nil {
		return
	}
	in := int(resp.UsageMetadata.PromptTokenCount)
	out := int(resp.UsageMetadata.CandidatesTokenCount)
	cacheRead := int(resp.UsageMetadata.CachedContentTokenCount)
	ev := UsageEvent{
		CallerID:             MeterCallerIDFromCtx(ctx),
		Provider:             "gemini",
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
	p.meter(ev)
}

func extractGeminiText(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 {
		return ""
	}
	candidate := resp.Candidates[0]
	if candidate.Content == nil {
		return ""
	}
	var sb strings.Builder
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// structToGeminiSchema reflects a Go struct into a genai.Schema for structured output.
func structToGeminiSchema(v any) (*genai.Schema, error) {
	b, err := json.Marshal(schemaFromStruct(v))
	if err != nil {
		return nil, err
	}
	var s genai.Schema
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// schemaFromStruct uses the jsonschema reflector to get a JSON Schema, then
// resolves $refs for Gemini compatibility.
func schemaFromStruct(v any) map[string]any {
	reflector := jsonschema.Reflector{}
	schema := reflector.Reflect(v)
	b, _ := json.Marshal(schema)
	var raw map[string]any
	json.Unmarshal(b, &raw) //nolint:errcheck
	defs, _ := raw["$defs"].(map[string]any)
	resolved := resolveRefs(raw, defs).(map[string]any)
	delete(resolved, "$defs")
	delete(resolved, "$schema")
	delete(resolved, "$id")
	return resolved
}
