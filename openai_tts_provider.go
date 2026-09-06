package ai

import (
	"context"
	"fmt"
	"io"
	"unicode/utf8"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

// OpenAI caps a single speech request at 4096 characters.
const openAITTSMaxChars = 4096

const openAITTSDefaultVoice = openai.VoiceCoral

type OpenAITTSProvider struct {
	raw   *openai.Client
	model string
	meter MeterHook
}

func newOpenAITTSProvider(apiKey, model string) *OpenAITTSProvider {
	if model == "" {
		model = string(openai.TTSModelGPT4oMini)
	}
	return &OpenAITTSProvider{raw: openai.NewClient(apiKey), model: model}
}

func (p *OpenAITTSProvider) SetMeter(hook MeterHook) { p.meter = hook }

func (p *OpenAITTSProvider) Synthesize(ctx context.Context, req TTSRequest) (io.ReadCloser, error) {
	if req.Text == "" {
		return nil, fmt.Errorf("openai tts: empty input")
	}
	chars := utf8.RuneCountInString(req.Text)
	if chars > openAITTSMaxChars {
		return nil, fmt.Errorf("openai tts: input is %d chars, max %d", chars, openAITTSMaxChars)
	}
	voice := openai.SpeechVoice(req.Voice)
	if voice == "" {
		voice = openAITTSDefaultVoice
	}
	format := openai.SpeechResponseFormat(req.Format)
	if format == "" {
		format = openai.SpeechResponseFormatMp3
	}
	logger.Debug("tts synthesize", zap.String("model", p.model), zap.String("voice", string(voice)), zap.String("format", string(format)), zap.Int("chars", chars))
	resp, err := p.raw.CreateSpeech(ctx, openai.CreateSpeechRequest{
		Model:          openai.SpeechModel(p.model),
		Input:          req.Text,
		Voice:          voice,
		ResponseFormat: format,
		Speed:          req.Speed,
		Instructions:   req.Instructions,
	})
	if err != nil {
		return nil, fmt.Errorf("openai tts: %w", err)
	}
	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openai",
			Model:            p.model,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      chars,
			TotalTokens:      chars,
			EstimatedCostUSD: EstimateTTSCost(p.model, chars),
			Metadata:         mergeMeterMetadata(ctx, map[string]any{"type": "tts", "voice": string(voice), "format": string(format)}),
		})
	}
	return resp, nil
}
