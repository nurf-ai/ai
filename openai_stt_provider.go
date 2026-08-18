package ai

import (
	"context"
	"fmt"
	"io"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

type OpenAISTTProvider struct {
	raw   *openai.Client
	model string
}

func newOpenAISTTProvider(apiKey, model string) *OpenAISTTProvider {
	if model == "" {
		model = openai.Whisper1
	}
	return &OpenAISTTProvider{raw: openai.NewClient(apiKey), model: model}
}

func (p *OpenAISTTProvider) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	logger.Debug("stt transcribe", zap.String("model", p.model), zap.String("filename", filename))
	resp, err := p.raw.CreateTranscription(ctx, openai.AudioRequest{
		Model:    p.model,
		Reader:   audio,
		FilePath: filename,
	})
	if err != nil {
		return "", fmt.Errorf("openai stt: %w", err)
	}
	return resp.Text, nil
}
