package ai

import (
	"context"
	"encoding/base64"
	"fmt"

	"go.uber.org/zap"
	"google.golang.org/genai"
)

const defaultGeminiImageModel = "gemini-2.5-flash-image"

// GeminiImageProvider implements ImageProvider using Google's Gemini API.
type GeminiImageProvider struct {
	client     *genai.Client
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func (p *GeminiImageProvider) WithModeration(m ModerationProvider) *GeminiImageProvider {
	p.moderation = m
	return p
}

func (p *GeminiImageProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *GeminiImageProvider) SetModeration(m ModerationProvider) { p.moderation = m }

func newGeminiImageProvider(ctx context.Context, apiKey, model string) (*GeminiImageProvider, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}
	if model == "" {
		model = defaultGeminiImageModel
	}
	return &GeminiImageProvider{client: client, model: model}, nil
}

func (p *GeminiImageProvider) Generate(ctx context.Context, prompt, model, _ string) (string, error) {
	if err := checkModeration(ctx, p.moderation, prompt); err != nil {
		return "", err
	}
	if model == "" {
		model = p.model
	}

	if TransparentBGFromCtx(ctx) {
		prompt += " The background MUST be a perfectly uniform solid bright green (#00FF00) with no gradients, shadows, or variations."
	}
	logger.Debug("generating", zap.String("model", model))

	contents := []*genai.Content{
		{Parts: []*genai.Part{genai.NewPartFromText(prompt)}, Role: genai.RoleUser},
	}
	resp, err := p.client.Models.GenerateContent(ctx, model, contents,
		&genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE", "TEXT"},
		},
	)
	if err != nil {
		return "", fmt.Errorf("gemini image generate: %w", err)
	}
	img, err := p.extractImage(ctx, resp)
	if err == nil {
		p.emitImageUsage(ctx)
	}
	return img, err
}

func (p *GeminiImageProvider) Edit(ctx context.Context, image []byte, editPrompt string) (string, error) {
	if err := checkModeration(ctx, p.moderation, editPrompt); err != nil {
		return "", err
	}
	if TransparentBGFromCtx(ctx) {
		editPrompt += " The background MUST be a perfectly uniform solid bright green (#00FF00) with no gradients, shadows, or variations."
	}
	logger.Debug("editing", zap.String("model", p.model), zap.Int("img_len", len(image)))
	logger.Log(traceLevel, "edit prompt", zap.String("prompt", editPrompt))

	contents := []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText(editPrompt),
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: image}},
			},
		},
	}
	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents,
		&genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE", "TEXT"},
		},
	)
	if err != nil {
		return "", fmt.Errorf("gemini image edit: %w", err)
	}
	img, err := p.extractImage(ctx, resp)
	if err == nil {
		p.emitImageUsage(ctx)
	}
	return img, err
}

func (p *GeminiImageProvider) EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) {
	if err := checkModeration(ctx, p.moderation, editPrompt); err != nil {
		return "", err
	}
	if TransparentBGFromCtx(ctx) {
		editPrompt += " The background MUST be a perfectly uniform solid bright green (#00FF00) with no gradients, shadows, or variations."
	}
	logger.Debug("edit-with-reference", zap.String("model", p.model), zap.Int("img_len", len(image)), zap.Int("ref_len", len(reference)))
	logger.Log(traceLevel, "edit-with-reference prompt", zap.String("prompt", editPrompt))

	contents := []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText(editPrompt),
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: image}},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: reference}},
			},
		},
	}
	resp, err := p.client.Models.GenerateContent(ctx, p.model, contents,
		&genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE", "TEXT"},
		},
	)
	if err != nil {
		return "", fmt.Errorf("gemini image edit-with-ref: %w", err)
	}
	img, err := p.extractImage(ctx, resp)
	if err == nil {
		p.emitImageUsage(ctx)
	}
	return img, err
}

func (p *GeminiImageProvider) emitImageUsage(ctx context.Context) {
	if p.meter == nil {
		return
	}
	p.meter(UsageEvent{
		CallerID:         MeterCallerIDFromCtx(ctx),
		Provider:         "gemini",
		Model:            p.model,
		Operation:        MeterOperationFromCtx(ctx),
		EstimatedCostUSD: EstimateCostFull(p.model, 0, 0, 0, 0),
		Metadata:         map[string]any{"type": "image_gen"},
	})
}

func (p *GeminiImageProvider) extractImage(ctx context.Context, resp *genai.GenerateContentResponse) (string, error) {
	if resp == nil || len(resp.Candidates) == 0 {
		return "", fmt.Errorf("gemini: no candidates returned")
	}
	candidate := resp.Candidates[0]
	if candidate.FinishReason != "" {
		logger.Log(traceLevel, "finish reason", zap.Any("reason", candidate.FinishReason))
	}
	for _, part := range candidate.Content.Parts {
		if part.InlineData != nil && part.InlineData.Data != nil {
			b64 := base64.StdEncoding.EncodeToString(part.InlineData.Data)
			logger.Debug("ok", zap.Int("b64_len", len(b64)))
			return b64, nil
		}
		if part.Text != "" {
			logger.Warn("got text instead of image", zap.String("text", part.Text))
		}
	}
	return "", fmt.Errorf("gemini: no image data in response")
}
