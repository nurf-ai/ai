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

const defaultTypesafeBase = "https://api.typesafe.ai"

// TypesafeJudgmentProvider implements JudgmentProvider using the Typesafe
// System One API (Jev model).
type TypesafeJudgmentProvider struct {
	apiKey string
	model  string
	base   string
	http   *http.Client
	meter  MeterHook
}

// TypesafeOption configures a TypesafeJudgmentProvider.
type TypesafeOption func(*TypesafeJudgmentProvider)

// WithTypesafeHTTPClient overrides the HTTP client.
func WithTypesafeHTTPClient(h *http.Client) TypesafeOption {
	return func(p *TypesafeJudgmentProvider) {
		if h != nil {
			p.http = h
		}
	}
}

// WithTypesafeBase overrides the API base URL.
func WithTypesafeBase(base string) TypesafeOption {
	return func(p *TypesafeJudgmentProvider) {
		if base != "" {
			p.base = strings.TrimRight(base, "/")
		}
	}
}

// WithTypesafeModel overrides the model (default "jev-latest").
func WithTypesafeModel(model string) TypesafeOption {
	return func(p *TypesafeJudgmentProvider) {
		if model != "" {
			p.model = model
		}
	}
}

func NewTypesafeJudgmentProvider(apiKey string, opts ...TypesafeOption) *TypesafeJudgmentProvider {
	p := &TypesafeJudgmentProvider{
		apiKey: apiKey,
		model:  "jev-latest",
		base:   defaultTypesafeBase,
		http:   &http.Client{},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *TypesafeJudgmentProvider) SetMeter(hook MeterHook) { p.meter = hook }

type typesafeRequest struct {
	State     any                         `json:"state"`
	Model     string                      `json:"model"`
	Questions map[string]JudgmentQuestion `json:"questions"`
}

func (p *TypesafeJudgmentProvider) Judge(ctx context.Context, req *JudgmentRequest) (*JudgmentResult, error) {
	wireReq := typesafeRequest{
		State:     req.State,
		Model:     p.model,
		Questions: req.Questions,
	}
	body, err := json.Marshal(wireReq)
	if err != nil {
		return nil, fmt.Errorf("typesafe: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.base+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("typesafe: build request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	resp, err := p.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("typesafe: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("typesafe: read body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &TypesafeError{
			Status:  resp.StatusCode,
			Message: typesafeErrorMessage(data),
		}
	}

	var result JudgmentResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("typesafe: decode response: %w", err)
	}

	if p.meter != nil {
		ev := UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "typesafe",
			Model:            result.Model,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      result.Usage.InputTokens,
			OutputTokens:     result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
			Metadata:         mergeMeterMetadata(ctx, nil),
		}
		attachBlocks(ctx, &ev)
		p.meter(ev)
	}

	logger.Debug("typesafe judge",
		zap.String("model", result.Model),
		zap.Int("input_tokens", result.Usage.InputTokens),
		zap.Int("output_tokens", result.Usage.OutputTokens),
		zap.Int("questions", len(result.Answers)),
	)

	return &result, nil
}

// TypesafeError is a non-2xx response from the Typesafe API.
type TypesafeError struct {
	Status  int
	Message string
}

func (e *TypesafeError) Error() string {
	return fmt.Sprintf("typesafe: status %d: %s", e.Status, e.Message)
}

func typesafeErrorMessage(data []byte) string {
	var env struct {
		Error   string `json:"error"`
		Message string `json:"message"`
		Detail  string `json:"detail"`
	}
	if json.Unmarshal(data, &env) == nil {
		for _, s := range []string{env.Error, env.Message, env.Detail} {
			if s != "" {
				return s
			}
		}
	}
	msg := strings.TrimSpace(string(data))
	if msg == "" {
		return "no error detail"
	}
	return truncate(msg, 300)
}
