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
	DefaultVeoVideoModel = "veo-3.1-fast-generate-preview"
	veoBaseURL           = "https://generativelanguage.googleapis.com/v1beta/models"
	veoPollInterval      = 10 * time.Second
)

type VeoVideoProvider struct {
	apiKey     string
	model      string
	httpClient *http.Client
	moderation ModerationProvider
	meter      MeterHook
}

func newVeoVideoProvider(apiKey, model string) *VeoVideoProvider {
	if model == "" {
		model = DefaultVeoVideoModel
	}
	return &VeoVideoProvider{
		apiKey:     apiKey,
		model:      model,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (p *VeoVideoProvider) Name() string                       { return "veo" }
func (p *VeoVideoProvider) Model() string                      { return p.model }
func (p *VeoVideoProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *VeoVideoProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type veoPredictReq struct {
	Instances  []veoInstance  `json:"instances"`
	Parameters veoParameters `json:"parameters"`
}

type veoInstance struct {
	Prompt string       `json:"prompt"`
	Image  *veoImage    `json:"image,omitempty"`
}

type veoImage struct {
	InlineData *veoInlineData `json:"inlineData"`
}

type veoInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type veoParameters struct {
	AspectRatio      string `json:"aspectRatio,omitempty"`
	Resolution       string `json:"resolution,omitempty"`
	DurationSeconds  string `json:"durationSeconds,omitempty"`
	PersonGeneration string `json:"personGeneration,omitempty"`
}

type veoOperationResp struct {
	Name     string          `json:"name"`
	Done     bool            `json:"done"`
	Response json.RawMessage `json:"response,omitempty"`
	Error    *veoError       `json:"error,omitempty"`
}

type veoError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type veoGenerateResp struct {
	GeneratedSamples []veoSample `json:"generatedSamples"`
}

type veoSample struct {
	Video veoVideo `json:"video"`
}

type veoVideo struct {
	URI string `json:"uri"`
}

func (p *VeoVideoProvider) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
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
		return nil, fmt.Errorf("veo video: marshal: %w", err)
	}

	logger.Debug("veo video submit", zap.String("model", model),
		zap.String("resolution", req.Resolution), zap.Bool("i2v", req.ImageURL != "" || len(req.Image) > 0))

	start := time.Now()
	opName, err := p.submit(ctx, model, payload)
	if err != nil {
		return nil, err
	}

	logger.Debug("veo video polling", zap.String("operation", opName))

	videoURL, err := p.poll(ctx, opName)
	if err != nil {
		return nil, err
	}

	duration := req.Duration
	if duration <= 0 {
		duration = 8
	}
	resolution := req.Resolution
	if resolution == "" {
		resolution = "720p"
	}

	res := &VideoResult{
		URL:         videoURL,
		ContentType: "video/mp4",
		Duration:    duration,
		Model:       model,
		Elapsed:     time.Since(start),
	}
	res.CostUSD = EstimateVideoCost(model, res.Duration, resolution)

	logger.Debug("veo video generated", zap.String("model", model),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed),
		zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "veo",
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

func (p *VeoVideoProvider) submit(ctx context.Context, model string, payload []byte) (string, error) {
	url := veoBaseURL + "/" + model + ":predictLongRunning"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("veo video: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("veo video: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("veo video: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("veo video: HTTP %d: %s", resp.StatusCode, body)
	}

	var op veoOperationResp
	if err := json.Unmarshal(body, &op); err != nil {
		return "", fmt.Errorf("veo video: decode submit: %w", err)
	}
	if op.Name == "" {
		return "", fmt.Errorf("veo video: empty operation name: %s", body)
	}
	return op.Name, nil
}

func (p *VeoVideoProvider) poll(ctx context.Context, opName string) (string, error) {
	url := "https://generativelanguage.googleapis.com/v1beta/" + opName
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(veoPollInterval):
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return "", fmt.Errorf("veo video poll: %w", err)
		}
		httpReq.Header.Set("x-goog-api-key", p.apiKey)

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			return "", fmt.Errorf("veo video poll: %w", err)
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return "", fmt.Errorf("veo video poll: read: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("veo video poll: HTTP %d: %s", resp.StatusCode, body)
		}

		var op veoOperationResp
		if err := json.Unmarshal(body, &op); err != nil {
			return "", fmt.Errorf("veo video poll: decode: %w", err)
		}

		if op.Error != nil {
			return "", fmt.Errorf("veo video: operation error %d: %s", op.Error.Code, op.Error.Message)
		}
		if !op.Done {
			logger.Debug("veo video poll", zap.String("operation", opName), zap.Bool("done", false))
			continue
		}

		var genResp veoGenerateResp
		if err := json.Unmarshal(op.Response, &genResp); err != nil {
			return "", fmt.Errorf("veo video: decode result: %w", err)
		}
		if len(genResp.GeneratedSamples) == 0 || genResp.GeneratedSamples[0].Video.URI == "" {
			return "", fmt.Errorf("veo video: no video in completed operation")
		}
		return genResp.GeneratedSamples[0].Video.URI, nil
	}
}

func (p *VeoVideoProvider) buildRequest(model string, req VideoRequest) veoPredictReq {
	inst := veoInstance{Prompt: req.Prompt}

	hasImage := req.ImageURL != "" || len(req.Image) > 0
	if hasImage {
		var imgData string
		mt := req.ImageMediaType
		if mt == "" {
			mt = "image/jpeg"
		}
		if len(req.Image) > 0 {
			imgData = base64.StdEncoding.EncodeToString(req.Image)
		}
		if imgData != "" {
			inst.Image = &veoImage{InlineData: &veoInlineData{MimeType: mt, Data: imgData}}
		}
	}

	params := veoParameters{PersonGeneration: "allow_all"}
	if req.AspectRatio != "" {
		params.AspectRatio = req.AspectRatio
	}
	if req.Resolution != "" {
		params.Resolution = req.Resolution
	}
	if req.Duration > 0 {
		params.DurationSeconds = fmt.Sprintf("%.0f", req.Duration)
	}

	return veoPredictReq{
		Instances:  []veoInstance{inst},
		Parameters: params,
	}
}
