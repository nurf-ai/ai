package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"go.uber.org/zap"
)

const DefaultFalAudioModel = "sonilo/v1.1/text-to-sound-effects"

// FalAudioProvider implements AudioProvider on top of fal.ai's Sonilo model.
type FalAudioProvider struct {
	client *FalClient
	model  string
	meter  MeterHook
}

func NewFalAudioProvider(apiKey, model string, opts ...FalOption) *FalAudioProvider {
	if model == "" {
		model = DefaultFalAudioModel
	}
	return &FalAudioProvider{client: NewFalClient(apiKey, opts...), model: model}
}

func (p *FalAudioProvider) Name() string            { return "fal" }
func (p *FalAudioProvider) Model() string            { return p.model }
func (p *FalAudioProvider) SetMeter(hook MeterHook)  { p.meter = hook }

type falAudioOutput struct {
	Audio falFile `json:"audio"`
}

func (p *FalAudioProvider) Generate(ctx context.Context, req AudioRequest) (*AudioResult, error) {
	if req.Prompt == "" {
		return nil, fmt.Errorf("fal audio: empty prompt")
	}
	endpoint := p.model

	input := map[string]any{"prompt": req.Prompt}
	if req.Duration > 0 {
		input["duration"] = req.Duration
	}
	if req.Format != "" {
		input["audio_format"] = req.Format
	}

	logger.Debug("fal audio generate", zap.String("endpoint", endpoint),
		zap.Int("duration", req.Duration), zap.String("format", req.Format))

	start := time.Now()
	raw, err := p.client.Run(ctx, endpoint, input)
	if err != nil {
		logger.Error("fal audio generate failed", zap.String("endpoint", endpoint), zap.Error(err))
		return nil, fmt.Errorf("fal audio generate: %w", err)
	}
	var out falAudioOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("fal audio generate: decode output: %w", err)
	}
	if out.Audio.URL == "" {
		return nil, fmt.Errorf("fal audio generate: no audio url in output")
	}

	dur := out.Audio.Duration
	if dur == 0 && req.Duration > 0 {
		dur = float64(req.Duration)
	}

	res := &AudioResult{
		URL:         out.Audio.URL,
		ContentType: out.Audio.ContentType,
		FileName:    out.Audio.FileName,
		FileSize:    out.Audio.FileSize,
		Duration:    dur,
		Model:       endpoint,
		Elapsed:     time.Since(start),
	}
	res.CostUSD = EstimateAudioCost(endpoint, res.Duration)

	logger.Debug("fal audio generated", zap.String("endpoint", endpoint),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed), zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "fal",
			Model:            endpoint,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: res.CostUSD,
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":       "audio_gen",
				"seconds":    res.Duration,
				"format":     req.Format,
				"elapsed_ms": res.Elapsed.Milliseconds(),
			}),
		})
	}
	return res, nil
}
