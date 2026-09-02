package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"
)

const (
	DefaultGeminiVideoModel = "gemini-omni-1.1-flash"
	geminiInteractionsURL   = "https://generativelanguage.googleapis.com/v1beta/interactions"
)

// GeminiVideoProvider implements VideoProvider via the Gemini Interactions API.
type GeminiVideoProvider struct {
	apiKey     string
	model      string
	httpClient *http.Client
	moderation ModerationProvider
	meter      MeterHook
}

func newGeminiVideoProvider(apiKey, model string) *GeminiVideoProvider {
	if model == "" {
		model = DefaultGeminiVideoModel
	}
	return &GeminiVideoProvider{
		apiKey:     apiKey,
		model:      model,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}
}

func (p *GeminiVideoProvider) Name() string                       { return "gemini" }
func (p *GeminiVideoProvider) Model() string                      { return p.model }
func (p *GeminiVideoProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *GeminiVideoProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type geminiInteractionReq struct {
	Model          string              `json:"model"`
	Input          json.RawMessage     `json:"input"`
	ResponseFormat *geminiRespFormat   `json:"response_format,omitempty"`
	GenConfig      *geminiVideoGenConf `json:"generation_config,omitempty"`
}

type geminiRespFormat struct {
	Type        string `json:"type"`
	AspectRatio string `json:"aspect_ratio,omitempty"`
	Resolution  string `json:"resolution,omitempty"`
	Delivery    string `json:"delivery,omitempty"`
}

type geminiVideoGenConf struct {
	VideoConfig *geminiVideoConfig `json:"video_config,omitempty"`
}

type geminiVideoConfig struct {
	Task string `json:"task,omitempty"`
}

type geminiInteractionResp struct {
	ID     string       `json:"id"`
	Status string       `json:"status"`
	Steps  []geminiStep `json:"steps"`
}

type geminiStep struct {
	Type    string          `json:"type"`
	Content []geminiContent `json:"content"`
}

type geminiContent struct {
	Type     string `json:"type"`
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
	URI      string `json:"uri"`
}

func (p *GeminiVideoProvider) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
	if err := checkModeration(ctx, p.moderation, req.Prompt); err != nil {
		return nil, err
	}
	model := req.Model
	if model == "" {
		model = p.model
	}

	body := p.buildRequest(model, req)
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("gemini video: marshal: %w", err)
	}

	logger.Debug("gemini video generate", zap.String("model", model),
		zap.String("resolution", req.Resolution), zap.Bool("i2v", len(req.Image) > 0 || req.ImageURL != ""))

	start := time.Now()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		geminiInteractionsURL+"?key="+p.apiKey, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("gemini video: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini video: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("gemini video: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini video: HTTP %d: %s", resp.StatusCode, respBody)
	}

	var ir geminiInteractionResp
	if err := json.Unmarshal(respBody, &ir); err != nil {
		return nil, fmt.Errorf("gemini video: decode response: %w", err)
	}

	videoContent := findVideoContent(ir.Steps)
	if videoContent == nil {
		return nil, fmt.Errorf("gemini video: no video in response (status=%s)", ir.Status)
	}

	res := &VideoResult{
		ContentType: videoContent.MimeType,
		Model:       model,
		Elapsed:     time.Since(start),
	}
	if res.ContentType == "" {
		res.ContentType = "video/mp4"
	}

	if videoContent.URI != "" {
		res.URL = videoContent.URI
	} else if videoContent.Data != "" {
		res.URL = "data:" + res.ContentType + ";base64," + videoContent.Data
		if decoded, err := base64.StdEncoding.DecodeString(videoContent.Data); err == nil {
			res.FileSize = int64(len(decoded))
		}
	}

	if req.Duration > 0 {
		res.Duration = req.Duration
	} else {
		res.Duration = 5 // Gemini default ~5s clips
	}

	resolution := req.Resolution
	if resolution == "" {
		resolution = "720p"
	}
	res.CostUSD = EstimateVideoCost(model, res.Duration, resolution)

	logger.Debug("gemini video generated", zap.String("model", model),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed), zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "gemini",
			Model:            model,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: res.CostUSD,
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":       "video_gen",
				"seconds":    res.Duration,
				"resolution": resolution,
				"elapsed_ms": res.Elapsed.Milliseconds(),
			}),
		})
	}
	return res, nil
}

func (p *GeminiVideoProvider) buildRequest(model string, req VideoRequest) geminiInteractionReq {
	ir := geminiInteractionReq{
		Model: model,
		ResponseFormat: &geminiRespFormat{
			Type:     "video",
			Delivery: "uri",
		},
	}

	if req.AspectRatio != "" {
		ir.ResponseFormat.AspectRatio = req.AspectRatio
	}
	if req.Resolution != "" {
		ir.ResponseFormat.Resolution = req.Resolution
	}

	hasImage := req.ImageURL != "" || len(req.Image) > 0
	if hasImage {
		ir.GenConfig = &geminiVideoGenConf{
			VideoConfig: &geminiVideoConfig{Task: "image_to_video"},
		}
		parts := []map[string]any{}

		if req.ImageURL != "" {
			parts = append(parts, map[string]any{
				"type": "image", "url": req.ImageURL,
			})
		} else {
			mt := req.ImageMediaType
			if mt == "" {
				mt = "image/jpeg"
			}
			parts = append(parts, map[string]any{
				"type": "image", "data": base64.StdEncoding.EncodeToString(req.Image), "mime_type": mt,
			})
		}
		parts = append(parts, map[string]any{
			"type": "text", "text": req.Prompt,
		})
		ir.Input, _ = json.Marshal(parts)
	} else {
		ir.Input, _ = json.Marshal(req.Prompt)
	}

	return ir
}

func findVideoContent(steps []geminiStep) *geminiContent {
	for i := len(steps) - 1; i >= 0; i-- {
		if steps[i].Type != "model_output" {
			continue
		}
		for j := range steps[i].Content {
			if steps[i].Content[j].Type == "video" {
				return &steps[i].Content[j]
			}
		}
	}
	return nil
}
