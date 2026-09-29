package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"
)

const DefaultFalImageModel = "fal-ai/flux/schnell"

// FalImageProvider implements ImageProvider on top of fal.ai's queue API.
// The model string is the full fal endpoint id (e.g. "fal-ai/flux/schnell").
type FalImageProvider struct {
	client     *FalClient
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func NewFalImageProvider(apiKey, model string, opts ...FalOption) *FalImageProvider {
	if model == "" {
		model = DefaultFalImageModel
	}
	return &FalImageProvider{client: NewFalClient(apiKey, opts...), model: model}
}

func NewFalImageProviderWithClient(client *FalClient, model string) *FalImageProvider {
	if model == "" {
		model = DefaultFalImageModel
	}
	return &FalImageProvider{client: client, model: model}
}

func (p *FalImageProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *FalImageProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type falImageFile struct {
	URL         string `json:"url"`
	ContentType string `json:"content_type"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
}

type falImageOutput struct {
	Images []falImageFile `json:"images"`
	Seed   int64          `json:"seed"`
	Prompt string         `json:"prompt"`
}

func (p *FalImageProvider) Generate(ctx context.Context, prompt, model, size string) (string, error) {
	if err := checkModeration(ctx, p.moderation, prompt); err != nil {
		return "", err
	}
	endpoint := model
	if endpoint == "" {
		endpoint = p.model
	}

	input := map[string]any{
		"prompt":     prompt,
		"num_images": 1,
	}
	if size != "" {
		if w, h, ok := parseSize(size); ok {
			input["image_size"] = map[string]int{"width": w, "height": h}
		}
	}

	logger.Debug("fal image generate", zap.String("endpoint", endpoint))

	start := time.Now()
	raw, err := p.client.Run(ctx, endpoint, input)
	if err != nil {
		logger.Error("fal image generate failed", zap.String("endpoint", endpoint), zap.Error(err))
		return "", fmt.Errorf("fal image generate: %w", err)
	}
	var out falImageOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("fal image generate: decode output: %w", err)
	}
	if len(out.Images) == 0 || out.Images[0].URL == "" {
		return "", fmt.Errorf("fal image generate: no image in output")
	}

	b64, err := fetchImageAsBase64(ctx, out.Images[0].URL)
	if err != nil {
		return "", fmt.Errorf("fal image generate: fetch image: %w", err)
	}

	elapsed := time.Since(start)
	logger.Debug("fal image generated", zap.String("endpoint", endpoint),
		zap.Int("b64_len", len(b64)), zap.Duration("elapsed", elapsed))

	if p.meter != nil {
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "fal",
			Model:            endpoint,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: EstimateCostFull(endpoint, 0, 0, 0, 0),
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":       "image_gen",
				"elapsed_ms": elapsed.Milliseconds(),
			}),
		})
	}
	return b64, nil
}

func (p *FalImageProvider) Edit(ctx context.Context, image []byte, editPrompt string) (string, error) {
	return "", fmt.Errorf("fal image edit: not supported (use Generate with a prompt)")
}

func (p *FalImageProvider) EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) {
	return "", fmt.Errorf("fal image edit-with-reference: not supported")
}

func fetchImageAsBase64(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close() //nolint:errcheck
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d fetching image", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20))
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

func parseSize(s string) (w, h int, ok bool) {
	parts := strings.SplitN(s, "x", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}
	w, err1 := strconv.Atoi(parts[0])
	h, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil || w <= 0 || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}
