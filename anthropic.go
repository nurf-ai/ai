package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/invopop/jsonschema"
	"go.uber.org/zap"
)

type AnthropicProvider struct {
	client     anthropic.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func (p *AnthropicProvider) WithModeration(m ModerationProvider) *AnthropicProvider {
	p.moderation = m
	return p
}

func (p *AnthropicProvider) WithMeter(hook MeterHook) *AnthropicProvider {
	p.meter = hook
	return p
}

func (p *AnthropicProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *AnthropicProvider) SetModeration(m ModerationProvider) { p.moderation = m }

func NewAnthropicProvider(apiKey, model string) *AnthropicProvider {
	client := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &AnthropicProvider{
		client: client,
		model:  model,
	}
}

func (p *AnthropicProvider) Name() string {
	return "Anthropic"
}

func (p *AnthropicProvider) Model() string {
	return p.model
}

// MaxInputTokens returns the advertised input context window for p.model.
func (p *AnthropicProvider) MaxInputTokens() (int64, error) {
	return MaxInputTokensLLM("anthropic", p.model)
}

func (p *AnthropicProvider) CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error {
	if err := checkModeration(ctx, p.moderation, userPrompt); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output",
		zap.String("provider", "anthropic"),
		zap.String("model", p.model),
		zap.String("userPrompt", userPrompt),
		zap.String("outputType", fmt.Sprintf("%T", structuredOutput)),
		zap.String("sysPrompt", sysPrompt),
	)
	reflector := jsonschema.Reflector{}
	schema := reflector.Reflect(structuredOutput)

	// Marshal the full schema to JSON so we can resolve all $ref entries.
	// The Anthropic tool schema API does not support $ref, so nested types
	// must be inlined or the model will treat them as generic/string values.
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("marshal schema: %w", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(schemaBytes, &raw); err != nil {
		return fmt.Errorf("unmarshal schema: %w", err)
	}

	defs, _ := raw["$defs"].(map[string]any)
	resolved := resolveRefs(raw, defs).(map[string]any)

	props := resolved["properties"]
	// Anthropic API requires a non-empty input_schema on custom tools — the
	// ToolInputSchemaParam struct uses json:"omitzero" so a zero-valued
	// instance (Properties nil, Required nil) marshals as missing entirely,
	// producing `tools.0.custom.input_schema: Field required`. Generic
	// destinations like `*map[string]any` (used by the crystallizer) reflect
	// to a schema with NO `properties` key at all → props is nil here.
	// Coerce to an empty map so the struct serializes as
	// `{"type":"object","properties":{}}`, satisfying the API. The model
	// is then free to emit any JSON shape via the description prompt.
	if props == nil {
		props = map[string]any{}
	}
	var required []string
	if req, ok := resolved["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				required = append(required, s)
			}
		}
	}

	message, err := p.client.Messages.New(ctx, anthropic.MessageNewParams{
		MaxTokens: 4096,
		Model:     anthropic.Model(p.model),
		System:    p.buildSysBlocks(ctx, sysPrompt),
		Tools: []anthropic.ToolUnionParam{
			{
				OfTool: &anthropic.ToolParam{
					Name:        "structured_output",
					Description: anthropic.String("Extract structured output"),
					InputSchema: anthropic.ToolInputSchemaParam{
						Type:       "object",
						Properties: props,
						Required:   required,
					},
				},
			},
		},
		ToolChoice: anthropic.ToolChoiceParamOfTool("structured_output"),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userPrompt)),
		},
	})
	if err != nil {
		return fmt.Errorf("anthropic API call failed: %w", err)
	}
	p.emitUsage(ctx, message, sysPrompt, userPrompt)

	for _, block := range message.Content {
		if toolUse, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			if err := json.Unmarshal(toolUse.Input, structuredOutput); err != nil {
				return fmt.Errorf("failed to unmarshal tool input: %w", err)
			}
			return nil
		}
	}

	return fmt.Errorf("no structured output in response")
}

func (p *AnthropicProvider) CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	return p.CreateStructuredOutputFromParts(ctx, []Part{TextPart{Text: userPrompt}}, sysPrompt, schema)
}

// CreateStructuredOutputFromParts is CreateStructuredOutputFromSchema with a
// multimodal user turn (text + base64 images). Honours WithMaxTokens
// (default 4096): a vision-authored spec routinely needs more.
func (p *AnthropicProvider) CreateStructuredOutputFromParts(ctx context.Context, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	userText, _ := PartsText(parts)
	if err := checkModeration(ctx, p.moderation, userText); err != nil {
		return nil, err
	}
	logger.Log(traceLevel, "structured output from schema", zap.String("provider", "anthropic"), zap.String("model", p.model), zap.Int("parts", len(parts)))

	var schemaObj map[string]any
	if err := json.Unmarshal(schema, &schemaObj); err != nil {
		return nil, fmt.Errorf("invalid schema JSON: %w", err)
	}

	// Extract properties and required from the JSON Schema
	var props map[string]any
	var required []string
	if p, ok := schemaObj["properties"].(map[string]any); ok {
		props = p
	}
	if r, ok := schemaObj["required"].([]any); ok {
		for _, v := range r {
			if s, ok := v.(string); ok {
				required = append(required, s)
			}
		}
	}

	// Build ordered properties for Anthropic SDK
	propBytes, _ := json.Marshal(props)
	var propsParsed map[string]jsonschema.Schema
	json.Unmarshal(propBytes, &propsParsed) //nolint:errcheck

	orderedProps := jsonschema.NewProperties()
	for k, v := range propsParsed {
		orderedProps.Set(k, &v)
	}

	maxOut := MaxTokensFromCtx(ctx, 4096)
	message, err := p.client.Messages.New(ctx, anthropic.MessageNewParams{
		MaxTokens: int64(maxOut),
		Model:     anthropic.Model(p.model),
		System:    p.buildSysBlocks(ctx, sysPrompt),
		Tools: []anthropic.ToolUnionParam{
			{
				OfTool: &anthropic.ToolParam{
					Name:        "structured_output",
					Description: anthropic.String("Generate structured output matching the schema"),
					InputSchema: anthropic.ToolInputSchemaParam{
						Properties: orderedProps,
						Required:   required,
					},
				},
			},
		},
		ToolChoice: anthropic.ToolChoiceParamOfTool("structured_output"),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropicPartBlocks(parts)...),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic API call failed: %w", err)
	}
	p.emitUsage(ctx, message, sysPrompt, userText)

	if string(message.StopReason) == "max_tokens" {
		return nil, fmt.Errorf("structured output truncated at max_tokens=%d: shrink the schema/output or raise the budget with ai.WithMaxTokens", maxOut)
	}
	if string(message.StopReason) == "refusal" {
		return nil, fmt.Errorf("structured output stopped by the provider safety classifier (stop_reason=refusal): ask for paraphrases instead of verbatim quotes")
	}

	for _, block := range message.Content {
		if toolUse, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			var result map[string]any
			if err := json.Unmarshal(toolUse.Input, &result); err != nil {
				return nil, fmt.Errorf("unmarshal tool input (stop_reason=%s, input_len=%d): %w", message.StopReason, len(toolUse.Input), err)
			}
			return result, nil
		}
	}

	return nil, fmt.Errorf("no structured output in response")
}

// anthropicPartBlocks converts a multimodal turn into Anthropic content blocks.
func anthropicPartBlocks(parts []Part) []anthropic.ContentBlockParamUnion {
	blocks := make([]anthropic.ContentBlockParamUnion, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case ImagePart:
			blocks = append(blocks, anthropic.NewImageBlockBase64(v.MediaType, v.Data))
		case TextPart:
			blocks = append(blocks, anthropic.NewTextBlock(v.Text))
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, anthropic.NewTextBlock(""))
	}
	return blocks
}

func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
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
	logger.Log(traceLevel, "chat", zap.String("provider", "anthropic"), zap.String("model", p.model), zap.Int("messages", len(messages)), zap.Any("tools", toolNames))
	var sysBlocks []anthropic.TextBlockParam
	var apiMessages []anthropic.MessageParam

	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			block := anthropic.TextBlockParam{Text: m.Content}
			if m.CacheControl != nil && m.CacheControl.Type == "ephemeral" {
				block.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			sysBlocks = append(sysBlocks, block)

		case RoleUser:
			if len(m.Parts) > 0 {
				var blocks []anthropic.ContentBlockParamUnion
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						blocks = append(blocks, anthropic.NewImageBlockBase64(v.MediaType, v.Data))
					case TextPart:
						blocks = append(blocks, anthropic.NewTextBlock(v.Text))
					}
				}
				apiMessages = append(apiMessages, anthropic.NewUserMessage(blocks...))
			} else {
				apiMessages = append(apiMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			}

		case RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfText: &anthropic.TextBlockParam{Text: m.Content}})
			}
			for _, tc := range m.ToolCalls {
				raw := json.RawMessage(tc.Arguments)
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    tc.ID,
						Name:  tc.Name,
						Input: raw,
					},
				})
			}
			apiMessages = append(apiMessages, anthropic.MessageParam{Role: "assistant", Content: blocks})

		case RoleTool:
			apiMessages = append(apiMessages, anthropic.MessageParam{
				Role: "user",
				Content: []anthropic.ContentBlockParamUnion{
					{OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: m.ToolCallID,
						Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: m.Content}}},
					}},
				},
			})
		}
	}

	var apiTools []anthropic.ToolUnionParam
	for _, t := range tools {
		props := make(map[string]any)
		required := []string{}
		if p, ok := t.Parameters["properties"]; ok {
			if pm, ok := p.(map[string]any); ok {
				props = pm
			}
		}
		if r, ok := t.Parameters["required"]; ok {
			if rs, ok := r.([]string); ok {
				required = rs
			} else if ri, ok := r.([]any); ok {
				for _, v := range ri {
					if s, ok := v.(string); ok {
						required = append(required, s)
					}
				}
			}
		}

		propBytes, _ := json.Marshal(props)
		var propsParsed map[string]jsonschema.Schema
		json.Unmarshal(propBytes, &propsParsed) //nolint:errcheck

		orderedProps := jsonschema.NewProperties()
		for k, v := range propsParsed {
			orderedProps.Set(k, &v)
		}

		apiTools = append(apiTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: orderedProps,
					Required:   required,
				},
			},
		})
	}

	params := anthropic.MessageNewParams{
		MaxTokens: int64(MaxTokensFromCtx(ctx, 4096)),
		Model:     anthropic.Model(p.model),
		Messages:  apiMessages,
	}
	if effort := ReasoningEffortFromCtx(ctx); effort != "" {
		budget := anthropicThinkingBudget(effort)
		if budget > params.MaxTokens {
			params.MaxTokens = budget + params.MaxTokens
		}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
	}
	if len(sysBlocks) > 0 {
		// Honor ai.WithCacheSysPrompt by marking the LAST sys block as the
		// cacheable prefix boundary (Anthropic caches everything up through
		// a block with cache_control). Zero-impact when flag is unset.
		if CacheSysPromptFromCtx(ctx) {
			sysBlocks[len(sysBlocks)-1].CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
		params.System = sysBlocks
	}
	if len(apiTools) > 0 {
		params.Tools = apiTools
	}

	msg, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic chat failed: %w", err)
	}
	// Reconstruct sys/user prompts for debug capture.
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
	p.emitUsage(ctx, msg, sysText.String(), userText.String())

	resp := &Response{}
	for _, block := range msg.Content {
		switch v := block.AsAny().(type) {
		case anthropic.TextBlock:
			if resp.Content != "" {
				resp.Content += "\n"
			}
			resp.Content += v.Text
		case anthropic.ToolUseBlock:
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:        v.ID,
				Name:      v.Name,
				Arguments: json.RawMessage(v.Input),
			})
		}
	}

	return resp, nil
}

// anthropicThinkingBudget maps a reasoning effort level to a thinking budget in tokens.
func anthropicThinkingBudget(effort string) int64 {
	switch effort {
	case "low":
		return 1024
	case "medium":
		return 4096
	case "high":
		return 16384
	default:
		return 4096
	}
}

// buildSysBlocks returns the sys-prompt content blocks for this call,
// honoring ai.WithCacheSysPrompt(ctx). When the cache flag is set, the
// single sys block is marked with cache_control: ephemeral — Anthropic
// keeps the serialized block cached for ~5 min and charges ~10% on cache
// hits. Zero accuracy impact; large cost reduction on stable-prefix
// prompts (planner sys prompt is 6k+ chars in our baseline).
func (p *AnthropicProvider) buildSysBlocks(ctx context.Context, sysPrompt string) []anthropic.TextBlockParam {
	block := anthropic.TextBlockParam{Text: sysPrompt}
	if CacheSysPromptFromCtx(ctx) {
		block.CacheControl = anthropic.NewCacheControlEphemeralParam()
	}
	return []anthropic.TextBlockParam{block}
}

// CreateStructuredOutputBreakpointed satisfies router.CachedStructuredLLM
// by emitting two system blocks each marked with cache_control: ephemeral.
// The provider hashes the prefix up through each cache_control marker, so
// when sysPrompt + stableMid is byte-stable across turns we hit the bp2
// entry (everything stable cached); when only sysPrompt is stable (e.g.
// the candidate list changed) we hit bp1 (sysPrompt cached, stableMid
// reprocessed). One marker per change-rate tier — see
// .wiki/context/cache-breakpoints.md.
//
// dynamicTail is the per-call query and rides as the user message — never
// cached.
func (p *AnthropicProvider) CreateStructuredOutputBreakpointed(
	ctx context.Context,
	sysPrompt, stableMid, dynamicTail string,
	structuredOutput any,
) error {
	if err := checkModeration(ctx, p.moderation, dynamicTail); err != nil {
		return err
	}
	logger.Log(traceLevel, "structured output breakpointed",
		zap.String("provider", "anthropic"),
		zap.String("model", p.model),
		zap.String("outputType", fmt.Sprintf("%T", structuredOutput)),
		zap.String("surf", "breakpointed"),
		zap.String("sysPrompt", sysPrompt),
		zap.String("stableMid", stableMid),
		zap.String("dynamicTail", dynamicTail),
	)

	reflector := jsonschema.Reflector{}
	schema := reflector.Reflect(structuredOutput)
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return fmt.Errorf("marshal schema: %w", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(schemaBytes, &raw); err != nil {
		return fmt.Errorf("unmarshal schema: %w", err)
	}
	defs, _ := raw["$defs"].(map[string]any)
	resolved := resolveRefs(raw, defs).(map[string]any)
	props := resolved["properties"]
	// Coerce nil props → empty map so InputSchema serializes (see comment
	// at CreateStructuredOutput call site for full rationale).
	if props == nil {
		props = map[string]any{}
	}
	var required []string
	if req, ok := resolved["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				required = append(required, s)
			}
		}
	}

	// Two sys blocks, each marked. bp1 = end of sys; bp2 = end of mid.
	sysBlocks := []anthropic.TextBlockParam{
		{
			Text:         sysPrompt,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		},
		{
			Text:         stableMid,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		},
	}

	message, err := p.client.Messages.New(ctx, anthropic.MessageNewParams{
		MaxTokens: 4096,
		Model:     anthropic.Model(p.model),
		System:    sysBlocks,
		Tools: []anthropic.ToolUnionParam{
			{
				OfTool: &anthropic.ToolParam{
					Name:        "structured_output",
					Description: anthropic.String("Extract structured output"),
					InputSchema: anthropic.ToolInputSchemaParam{
						Type:       "object",
						Properties: props,
						Required:   required,
					},
				},
			},
		},
		ToolChoice: anthropic.ToolChoiceParamOfTool("structured_output"),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(dynamicTail)),
		},
	})
	if err != nil {
		return fmt.Errorf("anthropic API call failed: %w", err)
	}
	p.emitUsage(ctx, message, sysPrompt+"\n\n"+stableMid, dynamicTail)

	for _, block := range message.Content {
		if toolUse, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
			if err := json.Unmarshal(toolUse.Input, structuredOutput); err != nil {
				return fmt.Errorf("failed to unmarshal tool input: %w", err)
			}
			return nil
		}
	}
	return fmt.Errorf("no structured output in response")
}

func (p *AnthropicProvider) emitUsage(ctx context.Context, msg *anthropic.Message, sysPrompt, userPrompt string) {
	if p.meter == nil || msg == nil {
		return
	}
	in := int(msg.Usage.InputTokens)
	out := int(msg.Usage.OutputTokens)
	cacheCreate := int(msg.Usage.CacheCreationInputTokens)
	cacheRead := int(msg.Usage.CacheReadInputTokens)
	ev := UsageEvent{
		CallerID:                 MeterCallerIDFromCtx(ctx),
		Provider:                 "anthropic",
		Model:                    p.model,
		Operation:                MeterOperationFromCtx(ctx),
		InputTokens:              in,
		OutputTokens:             out,
		TotalTokens:              in + out,
		CacheCreationInputTokens: cacheCreate,
		CacheReadInputTokens:     cacheRead,
		EstimatedCostUSD:         EstimateCostFull(p.model, in, out, cacheCreate, cacheRead),
		SystemPrompt:             TruncatePromptForDebug(sysPrompt),
		UserPrompt:               TruncatePromptForDebug(userPrompt),
		DebugSpanID:              DebugSpanIDFromCtx(ctx),
	}
	attachBlocks(ctx, &ev)
	ev.Metadata = mergeMeterMetadata(ctx, ev.Metadata)
	p.meter(ev)
}

func (p *AnthropicProvider) ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error) {
	for _, m := range messages {
		if m.Role == RoleUser {
			if err := checkModeration(ctx, p.moderation, m.Content); err != nil {
				return nil, err
			}
		}
	}

	var sysBlocks []anthropic.TextBlockParam
	var apiMessages []anthropic.MessageParam

	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			block := anthropic.TextBlockParam{Text: m.Content}
			if m.CacheControl != nil && m.CacheControl.Type == "ephemeral" {
				block.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			sysBlocks = append(sysBlocks, block)
		case RoleUser:
			if len(m.Parts) > 0 {
				var blocks []anthropic.ContentBlockParamUnion
				for _, p := range m.Parts {
					switch v := p.(type) {
					case ImagePart:
						blocks = append(blocks, anthropic.NewImageBlockBase64(v.MediaType, v.Data))
					case TextPart:
						blocks = append(blocks, anthropic.NewTextBlock(v.Text))
					}
				}
				apiMessages = append(apiMessages, anthropic.NewUserMessage(blocks...))
			} else {
				apiMessages = append(apiMessages, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))
			}
		case RoleAssistant:
			var blocks []anthropic.ContentBlockParamUnion
			if m.Content != "" {
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfText: &anthropic.TextBlockParam{Text: m.Content}})
			}
			for _, tc := range m.ToolCalls {
				raw := json.RawMessage(tc.Arguments)
				blocks = append(blocks, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{ID: tc.ID, Name: tc.Name, Input: raw},
				})
			}
			apiMessages = append(apiMessages, anthropic.MessageParam{Role: "assistant", Content: blocks})
		case RoleTool:
			apiMessages = append(apiMessages, anthropic.MessageParam{
				Role: "user",
				Content: []anthropic.ContentBlockParamUnion{
					{OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: m.ToolCallID,
						Content:   []anthropic.ToolResultBlockParamContentUnion{{OfText: &anthropic.TextBlockParam{Text: m.Content}}},
					}},
				},
			})
		}
	}

	var apiTools []anthropic.ToolUnionParam
	for _, t := range tools {
		props := make(map[string]any)
		required := []string{}
		if p, ok := t.Parameters["properties"]; ok {
			if pm, ok := p.(map[string]any); ok {
				props = pm
			}
		}
		if r, ok := t.Parameters["required"]; ok {
			if rs, ok := r.([]string); ok {
				required = rs
			} else if ri, ok := r.([]any); ok {
				for _, v := range ri {
					if s, ok := v.(string); ok {
						required = append(required, s)
					}
				}
			}
		}
		propBytes, _ := json.Marshal(props)
		var propsParsed map[string]jsonschema.Schema
		json.Unmarshal(propBytes, &propsParsed) //nolint:errcheck
		orderedProps := jsonschema.NewProperties()
		for k, v := range propsParsed {
			orderedProps.Set(k, &v)
		}
		apiTools = append(apiTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name: t.Name, Description: anthropic.String(t.Description),
				InputSchema: anthropic.ToolInputSchemaParam{Properties: orderedProps, Required: required},
			},
		})
	}

	params := anthropic.MessageNewParams{
		MaxTokens: int64(MaxTokensFromCtx(ctx, 4096)),
		Model:     anthropic.Model(p.model),
		Messages:  apiMessages,
	}
	if effort := ReasoningEffortFromCtx(ctx); effort != "" {
		budget := anthropicThinkingBudget(effort)
		if budget > params.MaxTokens {
			params.MaxTokens = budget + params.MaxTokens
		}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
	}
	if len(sysBlocks) > 0 {
		if CacheSysPromptFromCtx(ctx) {
			sysBlocks[len(sysBlocks)-1].CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
		params.System = sysBlocks
	}
	if len(apiTools) > 0 {
		params.Tools = apiTools
	}

	stream := p.client.Messages.NewStreaming(ctx, params)
	defer stream.Close() //nolint:errcheck // fire-and-forget cleanup

	var content strings.Builder
	var toolCalls []ToolCall
	var currentToolID, currentToolName string
	var currentToolArgs strings.Builder
	var inToolUse bool
	var inputTokens, outputTokens, cacheCreate, cacheRead int64

	for stream.Next() {
		event := stream.Current()
		switch evt := event.AsAny().(type) {
		case anthropic.MessageStartEvent:
			inputTokens = evt.Message.Usage.InputTokens
			cacheCreate = evt.Message.Usage.CacheCreationInputTokens
			cacheRead = evt.Message.Usage.CacheReadInputTokens
		case anthropic.ContentBlockStartEvent:
			switch block := evt.ContentBlock.AsAny().(type) {
			case anthropic.ToolUseBlock:
				inToolUse = true
				currentToolID = block.ID
				currentToolName = block.Name
				currentToolArgs.Reset()
			}
		case anthropic.ContentBlockDeltaEvent:
			switch delta := evt.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				content.WriteString(delta.Text)
				if err := cb(StreamChunk{Text: delta.Text}); err != nil {
					resp := &Response{Content: content.String()}
					if len(toolCalls) > 0 {
						resp.ToolCalls = toolCalls
					}
					return resp, err
				}
			case anthropic.InputJSONDelta:
				currentToolArgs.WriteString(delta.PartialJSON)
				if err := cb(StreamChunk{ToolName: currentToolName, ToolArg: delta.PartialJSON}); err != nil {
					resp := &Response{Content: content.String()}
					if len(toolCalls) > 0 {
						resp.ToolCalls = toolCalls
					}
					return resp, err
				}
			}
		case anthropic.ContentBlockStopEvent:
			if inToolUse {
				toolCalls = append(toolCalls, ToolCall{
					ID: currentToolID, Name: currentToolName,
					Arguments: json.RawMessage(currentToolArgs.String()),
				})
				inToolUse = false
			}
		case anthropic.MessageDeltaEvent:
			outputTokens = evt.Usage.OutputTokens
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("anthropic stream: %w", err)
	}

	sysText, userText := extractPrompts(messages)
	p.emitUsage(ctx, &anthropic.Message{
		Usage: anthropic.Usage{
			InputTokens: inputTokens, OutputTokens: outputTokens,
			CacheCreationInputTokens: cacheCreate, CacheReadInputTokens: cacheRead,
		},
	}, sysText, userText)

	resp := &Response{Content: content.String()}
	if len(toolCalls) > 0 {
		resp.ToolCalls = toolCalls
	}
	return resp, nil
}

// resolveRefs recursively inlines all $ref entries using the provided $defs map.
// This is needed because the Anthropic tool schema API does not support JSON Schema $ref.
func resolveRefs(node any, defs map[string]any) any {
	switch v := node.(type) {
	case map[string]any:
		if ref, ok := v["$ref"].(string); ok {
			refName := strings.TrimPrefix(ref, "#/$defs/")
			if def, ok := defs[refName]; ok {
				return resolveRefs(def, defs)
			}
		}
		result := make(map[string]any, len(v))
		for k, val := range v {
			if k == "$defs" {
				continue
			}
			result[k] = resolveRefs(val, defs)
		}
		return result
	case []any:
		result := make([]any, len(v))
		for i, val := range v {
			result[i] = resolveRefs(val, defs)
		}
		return result
	default:
		return node
	}
}
