package ai

import (
	"bytes"
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

// OpenAIImageProvider implements ImageProvider using the OpenAI images API.
type OpenAIImageProvider struct {
	raw          *openai.Client
	defaultModel string
	moderation   ModerationProvider
	meter        MeterHook
}

func (p *OpenAIImageProvider) WithModeration(m ModerationProvider) *OpenAIImageProvider {
	p.moderation = m
	return p
}

func (p *OpenAIImageProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *OpenAIImageProvider) SetModeration(m ModerationProvider) { p.moderation = m }

func newOpenAIImageProvider(apiKey, model string) *OpenAIImageProvider {
	if model == "" {
		model = openai.CreateImageModelGptImage1
	}
	return &OpenAIImageProvider{raw: openai.NewClient(apiKey), defaultModel: model}
}

func (p *OpenAIImageProvider) Generate(ctx context.Context, prompt, model, size string) (string, error) {
	if err := checkModeration(ctx, p.moderation, prompt); err != nil {
		return "", err
	}
	if model == "" {
		model = p.defaultModel
	}
	if size == "" {
		size = openai.CreateImageSize1024x1024
	}
	bg := openai.CreateImageBackgroundOpaque
	if TransparentBGFromCtx(ctx) {
		bg = openai.CreateImageBackgroundTransparent
	}
	logger.Debug("generating", zap.String("model", model), zap.String("size", size), zap.Any("bg", bg))
	resp, err := p.raw.CreateImage(ctx, openai.ImageRequest{
		Prompt:       prompt,
		Model:        model,
		N:            1,
		Size:         size,
		Quality:      openai.CreateImageQualityLow,
		OutputFormat: openai.CreateImageOutputFormatPNG,
		Background:   bg,
	})
	if err != nil {
		logger.Error("generate failed", zap.String("model", model), zap.String("size", size), zap.Error(err))
		return "", fmt.Errorf("openai image generate: %w", err)
	}
	if len(resp.Data) == 0 {
		logger.Error("no data returned", zap.String("model", model))
		return "", fmt.Errorf("openai image generate: no data returned")
	}
	logger.Debug("generated ok", zap.String("model", model), zap.Int("b64_len", len(resp.Data[0].B64JSON)))
	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openai",
			Model:            model,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: EstimateCostFull(model, 0, 0, 0, 0),
			Metadata:         map[string]any{"type": "image_gen"},
		})
	}
	return resp.Data[0].B64JSON, nil
}

func (p *OpenAIImageProvider) Edit(ctx context.Context, image []byte, editPrompt string) (string, error) {
	if err := checkModeration(ctx, p.moderation, editPrompt); err != nil {
		return "", err
	}
	logger.Debug("editing", zap.String("model", p.defaultModel), zap.Int("img_len", len(image)))
	logger.Log(traceLevel, "edit prompt", zap.String("prompt", editPrompt))

	if len(image) == 0 {
		return p.Generate(ctx, editPrompt, "", "")
	}
	resp, err := p.raw.CreateEditImage(ctx, openai.ImageEditRequest{
		Image:          bytes.NewReader(image),
		Prompt:         editPrompt,
		Model:          p.defaultModel,
		N:              1,
		Size:           openai.CreateImageSize1024x1024,
		ResponseFormat: "b64_json",
	})
	if err != nil {
		return "", fmt.Errorf("openai image edit: %w", err)
	}
	if len(resp.Data) == 0 {
		return "", fmt.Errorf("openai image edit: no data returned")
	}
	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openai",
			Model:            p.defaultModel,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: EstimateCostFull(p.defaultModel, 0, 0, 0, 0),
			Metadata:         map[string]any{"type": "image_edit"},
		})
	}
	return resp.Data[0].B64JSON, nil
}

func (p *OpenAIImageProvider) EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) {
	logger.Debug("edit-with-reference", zap.String("model", p.defaultModel), zap.Int("img_len", len(image)), zap.Int("ref_len", len(reference)))
	logger.Log(traceLevel, "edit-with-reference prompt", zap.String("prompt", editPrompt))
	return "", fmt.Errorf("openai image edit-with-reference not implemented")
}
