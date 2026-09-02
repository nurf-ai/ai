package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"go.uber.org/zap"
)

// DefaultFalVideoModel is the fal endpoint used when none is configured:
// LTX-2.3 fast, image-to-video (accepts a first-frame image_url).
const DefaultFalVideoModel = "fal-ai/ltx-2.3/image-to-video/fast"

// FalVideoProvider implements VideoProvider on top of fal.ai's queue API.
//
// The model string is the full fal endpoint id
// (e.g. "fal-ai/ltx-2.3/image-to-video/fast"). Input keys follow the LTX
// family schema (prompt, image_url, duration, resolution, aspect_ratio, fps,
// generate_audio, seed, negative_prompt); zero-valued request fields are
// omitted so other fal video endpoints that share those names keep working.
type FalVideoProvider struct {
	client     *FalClient
	model      string
	moderation ModerationProvider
	meter      MeterHook
}

func newFalVideoProvider(apiKey, model string, opts ...FalOption) *FalVideoProvider {
	if model == "" {
		model = DefaultFalVideoModel
	}
	return &FalVideoProvider{client: NewFalClient(apiKey, opts...), model: model}
}

// NewFalVideoProviderWithClient builds a provider around an existing client
// (tests, shared clients).
func NewFalVideoProviderWithClient(client *FalClient, model string) *FalVideoProvider {
	if model == "" {
		model = DefaultFalVideoModel
	}
	return &FalVideoProvider{client: client, model: model}
}

func (p *FalVideoProvider) Name() string                       { return "fal" }
func (p *FalVideoProvider) Model() string                      { return p.model }
func (p *FalVideoProvider) Client() *FalClient                 { return p.client }
func (p *FalVideoProvider) SetMeter(hook MeterHook)            { p.meter = hook }
func (p *FalVideoProvider) SetModeration(m ModerationProvider) { p.moderation = m }

type falFile struct {
	URL         string  `json:"url"`
	ContentType string  `json:"content_type"`
	FileName    string  `json:"file_name"`
	FileSize    int64   `json:"file_size"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	FPS         float64 `json:"fps"`
	Duration    float64 `json:"duration"`
	NumFrames   int     `json:"num_frames"`
}

type falVideoOutput struct {
	Video falFile `json:"video"`
	Seed  int64   `json:"seed"`
}

// Generate runs one clip generation and blocks until fal returns the file.
func (p *FalVideoProvider) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
	if err := checkModeration(ctx, p.moderation, req.Prompt); err != nil {
		return nil, err
	}
	endpoint := req.Model
	if endpoint == "" {
		endpoint = p.model
	}
	if req.ImageURL == "" && len(req.Image) == 0 {
		endpoint = textToVideoEndpoint(endpoint)
	}
	input := p.buildInput(endpoint, req)

	logger.Debug("fal video generate", zap.String("endpoint", endpoint),
		zap.Float64("duration", req.Duration), zap.String("resolution", req.Resolution),
		zap.Bool("i2v", input["image_url"] != nil))

	start := time.Now()
	raw, err := p.client.Run(ctx, endpoint, input)
	if err != nil {
		logger.Error("fal video generate failed", zap.String("endpoint", endpoint), zap.Error(err))
		return nil, fmt.Errorf("fal video generate: %w", err)
	}
	var out falVideoOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("fal video generate: decode output: %w", err)
	}
	if out.Video.URL == "" {
		return nil, fmt.Errorf("fal video generate: no video url in output")
	}

	res := &VideoResult{
		URL:         out.Video.URL,
		ContentType: out.Video.ContentType,
		FileName:    out.Video.FileName,
		FileSize:    out.Video.FileSize,
		Width:       out.Video.Width,
		Height:      out.Video.Height,
		FPS:         out.Video.FPS,
		Duration:    out.Video.Duration,
		NumFrames:   out.Video.NumFrames,
		Model:       endpoint,
		Seed:        out.Seed,
		Elapsed:     time.Since(start),
	}
	if res.Duration == 0 && req.Duration > 0 {
		res.Duration = req.Duration
	}
	if res.ContentType == "" {
		res.ContentType = "video/mp4"
	}
	res.CostUSD = EstimateVideoCost(endpoint, res.Duration, req.Resolution)

	logger.Debug("fal video generated", zap.String("endpoint", endpoint),
		zap.Float64("seconds", res.Duration), zap.Duration("elapsed", res.Elapsed), zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "fal",
			Model:            endpoint,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: res.CostUSD,
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":       "video_gen",
				"seconds":    res.Duration,
				"resolution": req.Resolution,
				"elapsed_ms": res.Elapsed.Milliseconds(),
			}),
		})
	}
	return res, nil
}

// textToVideoEndpoint maps an image-to-video endpoint to its text-to-video
// sibling for calls without a conditioning frame: image-to-video endpoints
// reject a missing image_url (422), and LTX serves the same model under both
// ids. Endpoints without the marker pass through unchanged.
func textToVideoEndpoint(endpoint string) string {
	return strings.Replace(endpoint, "/image-to-video/", "/text-to-video/", 1)
}

func (p *FalVideoProvider) buildInput(endpoint string, req VideoRequest) map[string]any {
	input := map[string]any{"prompt": req.Prompt}
	imageURL := req.ImageURL
	if imageURL == "" && len(req.Image) > 0 {
		mt := req.ImageMediaType
		if mt == "" {
			mt = "image/jpeg"
		}
		imageURL = DataURI(mt, req.Image)
	}
	if imageURL != "" {
		input["image_url"] = imageURL
	}
	if req.NegativePrompt != "" {
		input["negative_prompt"] = req.NegativePrompt
	}
	if req.Duration > 0 {
		input["duration"] = int(math.Round(req.Duration))
	}
	if req.Resolution != "" {
		input["resolution"] = req.Resolution
	}
	if req.AspectRatio != "" {
		input["aspect_ratio"] = req.AspectRatio
	}
	if req.FPS > 0 {
		input["fps"] = req.FPS
	}
	input["generate_audio"] = req.Audio
	if req.Seed != nil {
		input["seed"] = *req.Seed
	}
	if strings.Contains(endpoint, "minimax/") {
		if _, ok := input["prompt_expansion_mode"]; !ok {
			input["prompt_expansion_mode"] = "balanced"
		}
		delete(input, "generate_audio")
	}
	// LTX-Video 0.9 family — counts frames, has no audio track:
	//   ltxv-13b-* : prompt, negative_prompt, resolution (480p|720p),
	//                aspect_ratio, num_frames, frame_rate, seed
	//   ltx-video  : prompt, negative_prompt, seed only (768×512, ~5s)
	if isLTXV13B(endpoint) || isLTXVideo09(endpoint) {
		fps := req.FPS
		if fps <= 0 {
			fps = 24
		}
		delete(input, "duration")
		delete(input, "generate_audio")
		delete(input, "fps")
		if isLTXV13B(endpoint) {
			if req.Duration > 0 {
				input["num_frames"] = int(math.Round(req.Duration*float64(fps))) + 1
			}
			input["frame_rate"] = fps
			if r, _ := input["resolution"].(string); r != "" && r != "480p" && r != "720p" {
				delete(input, "resolution") // endpoint default (720p)
			}
		} else {
			delete(input, "resolution")
			delete(input, "aspect_ratio")
		}
	}
	return input
}

func isLTXV13B(endpoint string) bool    { return strings.Contains(endpoint, "/ltxv-") }
func isLTXVideo09(endpoint string) bool { return strings.HasSuffix(endpoint, "/ltx-video") }
