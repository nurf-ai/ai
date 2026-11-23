package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

// OpenAIImageProvider implements ImageProvider using the OpenAI images API.
type OpenAIImageProvider struct {
	raw          *openai.Client
	apiKey       string
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
		model = openai.CreateImageModelGptImage2
	}
	return &OpenAIImageProvider{raw: openai.NewClient(apiKey), apiKey: apiKey, defaultModel: model}
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
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openai",
			Model:            model,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      resp.Usage.InputTokens,
			OutputTokens:     resp.Usage.OutputTokens,
			EstimatedCostUSD: estimateOpenAIImageCost(model, resp.Usage),
			Metadata:         mergeMeterMetadata(ctx, map[string]any{"type": "image_gen"}),
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
	var b64 string
	var usage openai.ImageResponseUsage
	if strings.HasPrefix(p.defaultModel, "gpt-image") {
		// gpt-image-* models reject the response_format param that the library sends
		// unconditionally, so we make a direct multipart request.
		var err error
		b64, usage, err = p.editDirect(ctx, image, editPrompt)
		if err != nil {
			return "", fmt.Errorf("openai image edit: %w", err)
		}
	} else {
		req := openai.ImageEditRequest{
			Image:          bytes.NewReader(image),
			Prompt:         editPrompt,
			Model:          p.defaultModel,
			N:              1,
			Size:           openai.CreateImageSize1024x1024,
			ResponseFormat: "b64_json",
		}
		resp, err := p.raw.CreateEditImage(ctx, req)
		if err != nil {
			return "", fmt.Errorf("openai image edit: %w", err)
		}
		if len(resp.Data) == 0 {
			return "", fmt.Errorf("openai image edit: no data returned")
		}
		b64 = resp.Data[0].B64JSON
		usage = resp.Usage
	}
	if p.meter != nil {
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "openai",
			Model:            p.defaultModel,
			Operation:        MeterOperationFromCtx(ctx),
			InputTokens:      usage.InputTokens,
			OutputTokens:     usage.OutputTokens,
			EstimatedCostUSD: estimateOpenAIImageCost(p.defaultModel, usage),
			Metadata:         mergeMeterMetadata(ctx, map[string]any{"type": "image_edit"}),
		})
	}
	return b64, nil
}

func (p *OpenAIImageProvider) editDirect(ctx context.Context, image []byte, prompt string) (string, openai.ImageResponseUsage, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	fw, err := w.CreatePart(textproto.MIMEHeader{
		"Content-Disposition": {`form-data; name="image[]"; filename="image.png"`},
		"Content-Type":        {"image/png"},
	})
	if err != nil {
		return "", openai.ImageResponseUsage{}, err
	}
	if _, err := fw.Write(image); err != nil {
		return "", openai.ImageResponseUsage{}, err
	}
	_ = w.WriteField("prompt", prompt)
	_ = w.WriteField("model", p.defaultModel)
	_ = w.WriteField("n", "1")
	_ = w.WriteField("size", openai.CreateImageSize1024x1024)
	if err := w.Close(); err != nil {
		return "", openai.ImageResponseUsage{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/images/edits", &body)
	if err != nil {
		return "", openai.ImageResponseUsage{}, err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", openai.ImageResponseUsage{}, err
	}
	defer resp.Body.Close() //nolint:errcheck

	var result struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
		Usage openai.ImageResponseUsage `json:"usage"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", openai.ImageResponseUsage{}, fmt.Errorf("decode response: %w", err)
	}
	if result.Error != nil {
		return "", openai.ImageResponseUsage{}, fmt.Errorf("status %d: %s", resp.StatusCode, result.Error.Message)
	}
	if len(result.Data) == 0 || result.Data[0].B64JSON == "" {
		return "", openai.ImageResponseUsage{}, fmt.Errorf("no image data in response")
	}
	return result.Data[0].B64JSON, result.Usage, nil
}

func (p *OpenAIImageProvider) EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) {
	logger.Debug("edit-with-reference", zap.String("model", p.defaultModel), zap.Int("img_len", len(image)), zap.Int("ref_len", len(reference)))
	logger.Log(traceLevel, "edit-with-reference prompt", zap.String("prompt", editPrompt))
	return "", fmt.Errorf("openai image edit-with-reference not implemented")
}

// estimateOpenAIImageCost prices a gpt-image call from the token usage the
// API reports: text and image input at their own rates, image output tokens
// at image_output_per_million. Without usage it falls back to flat_per_image.
func estimateOpenAIImageCost(model string, u openai.ImageResponseUsage) float64 {
	p, ok := pricingTable[model]
	if !ok {
		return 0
	}
	if u.InputTokens == 0 && u.OutputTokens == 0 {
		return p.FlatPerImage
	}
	imgIn := p.ImageInputPerMillion
	if imgIn == 0 {
		imgIn = p.InputPerMillion
	}
	text := u.InputTokens - u.InputTokensDetails.ImageTokens
	cost := float64(text)*p.InputPerMillion + float64(u.InputTokensDetails.ImageTokens)*imgIn
	return (cost + float64(u.OutputTokens)*p.ImageOutputPerMillion) / 1_000_000
}
