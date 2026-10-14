package ai

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"unicode/utf8"

	"go.uber.org/zap"
)

const (
	DefaultElevenLabsTTSModel = "eleven_flash_v2_5"
	// DefaultElevenLabsVoice is "George", a premade voice every account has.
	DefaultElevenLabsVoice = "JBFqnCBsd6RMkjVDRZzb"
)

// ElevenLabsTTSProvider synthesizes speech with POST /v1/text-to-speech.
// TTSRequest.Voice is an ElevenLabs voice id; Instructions is not sent
// (eleven_v3 and later take audio tags such as [whispers] inline in Text).
type ElevenLabsTTSProvider struct {
	client *elevenLabsClient
	model  string
	meter  MeterHook
}

func NewElevenLabsTTSProvider(apiKey, model string) *ElevenLabsTTSProvider {
	if model == "" {
		model = DefaultElevenLabsTTSModel
	}
	return &ElevenLabsTTSProvider{client: newElevenLabsClient(apiKey), model: model}
}

func (p *ElevenLabsTTSProvider) SetMeter(hook MeterHook) { p.meter = hook }

func (p *ElevenLabsTTSProvider) Synthesize(ctx context.Context, req TTSRequest) (io.ReadCloser, error) {
	if req.Text == "" {
		return nil, fmt.Errorf("elevenlabs tts: empty input")
	}
	format, err := elevenLabsOutputFormat(req.Format)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs tts: %w", err)
	}
	voice := req.Voice
	if voice == "" {
		voice = DefaultElevenLabsVoice
	}
	chars := utf8.RuneCountInString(req.Text)
	body := map[string]any{"text": req.Text, "model_id": p.model}
	if req.Speed > 0 {
		body["voice_settings"] = map[string]any{"speed": req.Speed}
	}
	logger.Debug("tts synthesize", zap.String("provider", "elevenlabs"), zap.String("model", p.model), zap.String("voice", voice), zap.String("format", format), zap.Int("chars", chars))
	resp, err := p.client.postJSON(ctx, "/v1/text-to-speech/"+url.PathEscape(voice)+"?output_format="+url.QueryEscape(format), body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs tts: %w", err)
	}
	if p.meter != nil {
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "elevenlabs",
			Model:            p.model,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      chars,
			TotalTokens:      chars,
			EstimatedCostUSD: EstimateTTSCost(p.model, chars),
			Metadata:         mergeMeterMetadata(ctx, map[string]any{"type": "tts", "voice": voice, "format": format}),
		})
	}
	return resp.Body, nil
}
