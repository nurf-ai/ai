package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"
)

const (
	DefaultOpenRouterImageModel = "black-forest-labs/flux.2-klein-4b"
	openRouterImageURL          = "https://openrouter.ai/api/v1/images"
)

// OpenRouterImageProvider implements ImageProvider via the OpenRouter unified
// image API (POST /api/v1/images → b64_json).
type OpenRouterImageProvider struct {
	apiKey     string
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func NewOpenRouterImageProvider(apiKey, model string) *OpenRouterImageProvider {
	if model == "" {
		model = DefaultOpenRouterImageModel
	}
	return &OpenRouterImageProvider{apiKey: apiKey, model: model}
}

func (p *OpenRouterImageProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *OpenRouterImageProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type openRouterImageResp struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
		URL     string `json:"url"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (p *OpenRouterImageProvider) Generate(ctx context.Context, prompt, model, size string) (string, error) {
	if err := checkModeration(ctx, p.moderation, prompt); err != nil {
		return "", err
	}
	if model == "" {
		model = p.model
	}

	logger.Debug("openrouter image generating", zap.String("model", model))

	body := map[string]any{
		"model":  model,
		"prompt": prompt,
		"n":      1,
	}
	if size != "" {
		body["resolution"] = openRouterResolution(size)
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("openrouter image: marshal: %w", err)
	}

	start := time.Now()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, openRouterImageURL, strings.NewReader(string(payload)))
	if err != nil {
		return "", fmt.Errorf("openrouter image: new request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("openrouter image: request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20))
	if err != nil {
		return "", fmt.Errorf("openrouter image: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("openrouter image: status %d: %s", resp.StatusCode, truncate(string(respBody), 500))
	}

	var result openRouterImageResp
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("openrouter image: decode: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("openrouter image: %s", result.Error.Message)
	}
	if len(result.Data) == 0 {
		return "", fmt.Errorf("openrouter image: no data returned")
	}

	b64 := result.Data[0].B64JSON
	if b64 == "" && result.Data[0].URL != "" {
		b64, err = fetchImageAsBase64(ctx, result.Data[0].URL)
		if err != nil {
			return "", fmt.Errorf("openrouter image: fetch url: %w", err)
		}
	}
	if b64 == "" {
		return "", fmt.Errorf("openrouter image: no image data in response")
	}

	elapsed := time.Since(start)
	logger.Debug("openrouter image ok", zap.String("model", model), zap.Int("b64_len", len(b64)), zap.Duration("elapsed", elapsed))

	if p.meter != nil {
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openrouter",
			Model:            model,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: EstimateCostFull(model, 0, 0, 0, 0),
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":       "image_gen",
				"elapsed_ms": elapsed.Milliseconds(),
			}),
		})
	}
	return b64, nil
}

func (p *OpenRouterImageProvider) Edit(ctx context.Context, image []byte, editPrompt string) (string, error) {
	return "", fmt.Errorf("openrouter image edit: not supported")
}

func (p *OpenRouterImageProvider) EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) {
	return "", fmt.Errorf("openrouter image edit-with-reference: not supported")
}

// openRouterResolution maps "WxH" or raw values to OpenRouter's enum:
// "512", "768", "1K", "2K", "4K".
func openRouterResolution(size string) string {
	w, _, ok := parseSize(size)
	if !ok {
		return size
	}
	switch {
	case w <= 512:
		return "512"
	case w <= 768:
		return "768"
	case w <= 1024:
		return "1K"
	case w <= 2048:
		return "2K"
	default:
		return "4K"
	}
}
