package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"
)

const (
	DefaultMinimaxVideoModel = "MiniMax-H3"
	minimaxVideoSubmitURL    = "https://api.minimax.io/v2/video_generation"
	minimaxVideoQueryURL     = "https://api.minimax.io/v2/query/video_generation"
	minimaxPollInterval      = 10 * time.Second
)

type MinimaxVideoProvider struct {
	apiKey     string
	model      string
	httpClient *http.Client
	moderation ModerationProvider
	meter      MeterHook
}

func newMinimaxVideoProvider(apiKey, model string) *MinimaxVideoProvider {
	if model == "" {
		model = DefaultMinimaxVideoModel
	}
	return &MinimaxVideoProvider{
		apiKey:     apiKey,
		model:      model,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (p *MinimaxVideoProvider) Name() string                       { return "minimax" }
func (p *MinimaxVideoProvider) Model() string                      { return p.model }
func (p *MinimaxVideoProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *MinimaxVideoProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type minimaxSubmitReq struct {
	Model      string           `json:"model"`
	Content    []minimaxContent `json:"content"`
	Duration   int              `json:"duration,omitempty"`
	Resolution string           `json:"resolution,omitempty"`
	Ratio      string           `json:"ratio,omitempty"`
}

type minimaxContent struct {
	Type     string           `json:"type"`
	Text     string           `json:"text,omitempty"`
	ImageURL *minimaxImageURL `json:"image_url,omitempty"`
	Role     string           `json:"role,omitempty"`
}

type minimaxImageURL struct {
	URL string `json:"url"`
}

type minimaxSubmitResp struct {
	TaskID string `json:"task_id"`
}

type minimaxQueryResp struct {
	Task minimaxTask `json:"task"`
}

type minimaxTask struct {
	TaskID  string              `json:"task_id"`
	Status  string              `json:"status"`
	Content *minimaxTaskContent `json:"content,omitempty"`
}

type minimaxTaskContent struct {
	URL string `json:"url"`
}

func (p *MinimaxVideoProvider) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
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
		return nil, fmt.Errorf("minimax video: marshal: %w", err)
	}

	logger.Debug("minimax video submit", zap.String("model", model),
		zap.String("resolution", req.Resolution), zap.Bool("i2v", req.ImageURL != "" || len(req.Image) > 0))

	start := time.Now()
	taskID, err := p.submit(ctx, payload)
	if err != nil {
		return nil, err
	}

	logger.Debug("minimax video polling", zap.String("task_id", taskID))

	videoURL, err := p.poll(ctx, taskID)
	if err != nil {
		return nil, err
	}

	duration := req.Duration
	if duration <= 0 {
		duration = 5
	}
	resolution := minimaxResolution(req.Resolution)
	if resolution == "" {
		resolution = "768P"
	}

	res := &VideoResult{
		URL:         videoURL,
		ContentType: "video/mp4",
		Duration:    duration,
		Model:       model,
		Elapsed:     time.Since(start),
	}
	res.CostUSD = EstimateVideoCost(minimaxPricingKey(model, req.ImageURL != "" || len(req.Image) > 0), res.Duration, resolution)

	logger.Debug("minimax video generated", zap.String("model", model),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed),
		zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "minimax",
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

func (p *MinimaxVideoProvider) submit(ctx context.Context, payload []byte) (string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, minimaxVideoSubmitURL, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("minimax video: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("minimax video: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("minimax video: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("minimax video: HTTP %d: %s", resp.StatusCode, body)
	}

	var sr minimaxSubmitResp
	if err := json.Unmarshal(body, &sr); err != nil {
		return "", fmt.Errorf("minimax video: decode submit: %w", err)
	}
	if sr.TaskID == "" {
		return "", fmt.Errorf("minimax video: empty task_id in submit response: %s", body)
	}
	return sr.TaskID, nil
}

func (p *MinimaxVideoProvider) poll(ctx context.Context, taskID string) (string, error) {
	url := minimaxVideoQueryURL + "/" + taskID
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(minimaxPollInterval):
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return "", fmt.Errorf("minimax video poll: %w", err)
		}
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			return "", fmt.Errorf("minimax video poll: %w", err)
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("minimax video poll: read: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("minimax video poll: HTTP %d: %s", resp.StatusCode, body)
		}

		var qr minimaxQueryResp
		if err := json.Unmarshal(body, &qr); err != nil {
			return "", fmt.Errorf("minimax video poll: decode: %w", err)
		}

		switch qr.Task.Status {
		case "succeeded":
			if qr.Task.Content == nil || qr.Task.Content.URL == "" {
				return "", fmt.Errorf("minimax video: succeeded but no video URL")
			}
			return qr.Task.Content.URL, nil
		case "failed", "cancelled":
			return "", fmt.Errorf("minimax video: task %s: %s", taskID, qr.Task.Status)
		default:
			logger.Debug("minimax video poll", zap.String("task_id", taskID), zap.String("status", qr.Task.Status))
		}
	}
}

func (p *MinimaxVideoProvider) buildRequest(model string, req VideoRequest) minimaxSubmitReq {
	r := minimaxSubmitReq{
		Model: model,
	}

	if req.Duration > 0 {
		r.Duration = int(req.Duration)
	}
	if req.Resolution != "" {
		r.Resolution = minimaxResolution(req.Resolution)
	}

	hasImage := req.ImageURL != "" || len(req.Image) > 0
	if hasImage {
		imageURL := req.ImageURL
		if imageURL == "" {
			mt := req.ImageMediaType
			if mt == "" {
				mt = "image/jpeg"
			}
			imageURL = DataURI(mt, req.Image)
		}
		r.Content = []minimaxContent{
			{Type: "text", Text: req.Prompt},
			{Type: "image_url", ImageURL: &minimaxImageURL{URL: imageURL}, Role: "first_frame"},
		}
	} else {
		r.Content = []minimaxContent{
			{Type: "text", Text: req.Prompt},
		}
		if req.AspectRatio != "" {
			r.Ratio = req.AspectRatio
		}
	}

	return r
}

// minimaxResolution maps a caller's resolution onto what MiniMax-H3 accepts
// (480P, 768P, 2K — a 1080p request is a 400 "does not support resolution").
// Picks the nearest tier at or below the requested height; unknown strings
// pass through upper-cased (MiniMax spells tiers 768P, not 768p).
func minimaxResolution(requested string) string {
	up := strings.ToUpper(strings.TrimSpace(requested))
	switch up {
	case "", "480P", "768P", "2K":
		return up
	}
	n := 0
	for _, c := range up {
		if c < '0' || c > '9' {
			break
		}
		n = n*10 + int(c-'0')
	}
	switch {
	case n == 0:
		return up
	case n >= 1440:
		return "2K"
	case n >= 768:
		return "768P"
	default:
		return "480P"
	}
}

// minimaxPricingKey returns the model ID as-is — models.json has direct
// entries under the "minimax" provider (MiniMax-H3, MiniMax-H3-Max).
func minimaxPricingKey(model string, _ bool) string { return model }
