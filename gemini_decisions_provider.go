package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"go.uber.org/zap"
)

const defaultGeminiDecisionsBase = "https://generativelanguage.googleapis.com/v1beta"

// GeminiDecisionsProvider implements DecisionsProvider using the Gemini
// Decisions API (gemini-decision-1 model).
type GeminiDecisionsProvider struct {
	apiKey string
	model  string
	base   string
	http   *http.Client
	meter  MeterHook
}

// GeminiDecisionsOption configures a GeminiDecisionsProvider.
type GeminiDecisionsOption func(*GeminiDecisionsProvider)

// WithGeminiDecisionsHTTPClient overrides the HTTP client.
func WithGeminiDecisionsHTTPClient(h *http.Client) GeminiDecisionsOption {
	return func(p *GeminiDecisionsProvider) {
		if h != nil {
			p.http = h
		}
	}
}

// WithGeminiDecisionsBase overrides the API base URL.
func WithGeminiDecisionsBase(base string) GeminiDecisionsOption {
	return func(p *GeminiDecisionsProvider) {
		if base != "" {
			p.base = strings.TrimRight(base, "/")
		}
	}
}

// WithGeminiDecisionsModel overrides the model (default "gemini-decision-1").
func WithGeminiDecisionsModel(model string) GeminiDecisionsOption {
	return func(p *GeminiDecisionsProvider) {
		if model != "" {
			p.model = model
		}
	}
}

func NewGeminiDecisionsProvider(apiKey string, opts ...GeminiDecisionsOption) *GeminiDecisionsProvider {
	p := &GeminiDecisionsProvider{
		apiKey: apiKey,
		model:  "gemini-decision-1",
		base:   defaultGeminiDecisionsBase,
		http:   &http.Client{},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *GeminiDecisionsProvider) SetMeter(hook MeterHook) { p.meter = hook }

type geminiDecisionsRequest struct {
	Input     any                          `json:"input"`
	Model     string                       `json:"model"`
	Questions map[string]DecisionsQuestion `json:"questions"`
}

func (p *GeminiDecisionsProvider) Judge(ctx context.Context, req *DecisionsRequest) (*DecisionsResult, error) {
	if req == nil {
		return nil, fmt.Errorf("gemini decisions: request is nil")
	}
	input := req.State
	switch input.(type) {
	case string, DecisionsContent, []DecisionsContent:
	default:
		data, err := json.Marshal(input)
		if err != nil {
			return nil, fmt.Errorf("gemini decisions: encode state: %w", err)
		}
		input = string(data)
	}
	questions := make(map[string]DecisionsQuestion, len(req.Questions))
	for key, question := range req.Questions {
		switch criteria := question.Criteria.(type) {
		case NoulCriteria:
			question.Criteria = map[string]string{"true_description": criteria.True, "false_description": criteria.False}
		case map[string]any:
			// Requests decoded from JSON have untyped criteria rather than
			// the concrete types returned by the question builders.
			converted := make(map[string]any, len(criteria))
			for key, value := range criteria {
				if question.Type == QuestionNoul {
					switch key {
					case "true":
						key = "true_description"
					case "false":
						key = "false_description"
					}
				}
				if question.Type == QuestionChoice && value == nil {
					value = ""
				}
				converted[key] = value
			}
			question.Criteria = converted
		case map[string]*string:
			options := make(map[string]string, len(criteria))
			for option, description := range criteria {
				if description == nil {
					options[option] = ""
				} else {
					options[option] = *description
				}
			}
			question.Criteria = options
		}
		questions[key] = question
	}
	wireReq := geminiDecisionsRequest{
		Input:     input,
		Model:     p.model,
		Questions: questions,
	}
	body, err := json.Marshal(wireReq)
	if err != nil {
		return nil, fmt.Errorf("gemini decisions: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/decisions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini decisions: build request: %w", err)
	}
	httpReq.Header.Set("x-goog-api-key", p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini decisions: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("gemini decisions: read body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &GeminiDecisionsError{
			Status:  resp.StatusCode,
			Message: geminiDecisionsErrorMessage(data),
		}
	}

	var response struct {
		Model   string                     `json:"model"`
		Answers map[string]DecisionsAnswer `json:"answers"`
		Usage   struct {
			InputTokens  int `json:"total_input_tokens"`
			OutputTokens int `json:"total_output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("gemini decisions: decode response: %w", err)
	}

	result := DecisionsResult{
		Model:   response.Model,
		Answers: response.Answers,
		Usage: DecisionsUsage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
		},
	}
	if result.Model == "" {
		result.Model = p.model
	}
	if p.meter != nil {
		ev := UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "gemini",
			Model:            result.Model,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      result.Usage.InputTokens,
			OutputTokens:     result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
			EstimatedCostUSD: EstimateCostFull(result.Model, result.Usage.InputTokens, result.Usage.OutputTokens, 0, 0),
			Metadata:         mergeMeterMetadata(ctx, nil),
		}
		attachBlocks(ctx, &ev)
		p.meter(ev)
	}

	logger.Debug("gemini decisions judge",
		zap.String("model", result.Model),
		zap.Int("input_tokens", result.Usage.InputTokens),
		zap.Int("output_tokens", result.Usage.OutputTokens),
		zap.Int("questions", len(result.Answers)),
	)

	return &result, nil
}

// GeminiDecisionsError is a non-2xx response from the Gemini Decisions API.
type GeminiDecisionsError struct {
	Status  int
	Message string
}

func (e *GeminiDecisionsError) Error() string {
	return fmt.Sprintf("gemini decisions: status %d: %s", e.Status, e.Message)
}

func geminiDecisionsErrorMessage(data []byte) string {
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &envelope) == nil && envelope.Error.Message != "" {
		return envelope.Error.Message
	}
	message := strings.TrimSpace(string(data))
	if message == "" {
		return "no error detail"
	}
	return truncate(message, 300)
}

var _ DecisionsProvider = (*GeminiDecisionsProvider)(nil)
var _ DecisionsMeterable = (*GeminiDecisionsProvider)(nil)
