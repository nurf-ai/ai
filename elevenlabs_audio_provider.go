package ai

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"time"

	"go.uber.org/zap"
)

const DefaultElevenLabsAudioModel = "eleven_text_to_sound_v2"

// ElevenLabsAudioProvider generates sound effects with POST
// /v1/sound-generation. The API answers with the audio bytes, so
// AudioResult.URL is a data: URI.
type ElevenLabsAudioProvider struct {
	client *elevenLabsClient
	model  string
	meter  MeterHook
}

func NewElevenLabsAudioProvider(apiKey, model string) *ElevenLabsAudioProvider {
	if model == "" {
		model = DefaultElevenLabsAudioModel
	}
	return &ElevenLabsAudioProvider{client: newElevenLabsClient(apiKey), model: model}
}

func (p *ElevenLabsAudioProvider) Name() string            { return "elevenlabs" }
func (p *ElevenLabsAudioProvider) Model() string           { return p.model }
func (p *ElevenLabsAudioProvider) SetMeter(hook MeterHook) { p.meter = hook }

// Generate makes one sound effect. Duration 0 lets the model pick the length
// (the API takes 0.5–30 s); Format is mp3 (default), pcm, opus or wav.
func (p *ElevenLabsAudioProvider) Generate(ctx context.Context, req AudioRequest) (*AudioResult, error) {
	if req.Prompt == "" {
		return nil, fmt.Errorf("elevenlabs audio: empty prompt")
	}
	format, err := elevenLabsOutputFormat(req.Format)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs audio: %w", err)
	}
	body := map[string]any{"text": req.Prompt, "model_id": p.model}
	if req.Duration > 0 {
		body["duration_seconds"] = req.Duration
	}
	logger.Debug("elevenlabs audio generate", zap.String("model", p.model), zap.Int("duration", req.Duration), zap.String("format", format))

	start := time.Now()
	resp, err := p.client.postJSON(ctx, "/v1/sound-generation?output_format="+url.QueryEscape(format), body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs audio generate: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck
	audio, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs audio generate: read: %w", err)
	}
	if len(audio) == 0 {
		return nil, fmt.Errorf("elevenlabs audio generate: empty audio")
	}

	contentType := resp.Header.Get("Content-Type")
	// the length asked for, else what the bytes at the format's bitrate hold
	dur := float64(req.Duration)
	estimated := false
	if dur == 0 {
		dur, estimated = elevenLabsAudioSeconds(format, int64(len(audio))), true
	}
	res := &AudioResult{
		URL:         DataURI(contentType, audio),
		ContentType: contentType,
		FileSize:    int64(len(audio)),
		Duration:    dur,
		Model:       p.model,
		Elapsed:     time.Since(start),
	}
	res.CostUSD = EstimateAudioCost(p.model, res.Duration)

	logger.Debug("elevenlabs audio generated", zap.String("model", p.model),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed), zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		md := map[string]any{
			"type":       "audio_gen",
			"seconds":    res.Duration,
			"format":     format,
			"elapsed_ms": res.Elapsed.Milliseconds(),
		}
		if estimated {
			md["estimated"] = true
		}
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "elevenlabs",
			Model:            p.model,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: res.CostUSD,
			Metadata:         mergeMeterMetadata(ctx, md),
		})
	}
	return res, nil
}
