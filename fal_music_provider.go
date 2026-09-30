package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"go.uber.org/zap"
)

// DefaultFalMusicModel writes full songs with vocals from one prompt.
const DefaultFalMusicModel = "google/lyria-3.5"

// FalMusicProvider implements MusicProvider on top of fal.ai's song models.
// Two input shapes are supported: MiniMax Music v2 endpoints take lyrics and
// an instrumental flag as fields; every other endpoint (Lyria) takes one
// prompt, so lyrics and the instrumental ask are folded into it.
type FalMusicProvider struct {
	client *FalClient
	model  string
	meter  MeterHook
}

func NewFalMusicProvider(apiKey, model string, opts ...FalOption) *FalMusicProvider {
	if model == "" {
		model = DefaultFalMusicModel
	}
	return &FalMusicProvider{client: NewFalClient(apiKey, opts...), model: model}
}

func (p *FalMusicProvider) Name() string            { return "fal" }
func (p *FalMusicProvider) Model() string           { return p.model }
func (p *FalMusicProvider) SetMeter(hook MeterHook) { p.meter = hook }

type falMusicOutput struct {
	Audio  falFile `json:"audio"`
	Lyrics string  `json:"lyrics"`
}

// falMusicInput maps a request onto the endpoint's input schema.
func falMusicInput(endpoint string, req MusicRequest) map[string]any {
	if strings.HasPrefix(endpoint, "fal-ai/minimax-music/v2") {
		in := map[string]any{"prompt": req.Prompt, "is_instrumental": req.Instrumental}
		switch {
		case req.Lyrics != "":
			in["lyrics"] = req.Lyrics
		case !req.Instrumental:
			in["lyrics_optimizer"] = true // no words given: the model writes them
		}
		return in
	}
	prompt := req.Prompt
	if req.Instrumental {
		prompt += "\n\nInstrumental only, no vocals."
	}
	if req.Lyrics != "" {
		prompt += "\n\nLyrics:\n" + req.Lyrics
	}
	return map[string]any{"prompt": prompt}
}

var lyricsSectionTag = regexp.MustCompile(`^\[\[[^\]]*\]\]$`)

// cleanLyrics turns Lyria's annotated lyrics ("[[B1]]" section markers,
// "[:] " line prefixes) into plain stanzas separated by blank lines.
func cleanLyrics(s string) string {
	var out []string
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if lyricsSectionTag.MatchString(line) {
			if len(out) > 0 && out[len(out)-1] != "" {
				out = append(out, "")
			}
			continue
		}
		out = append(out, strings.TrimSpace(strings.TrimPrefix(line, "[:]")))
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

func (p *FalMusicProvider) Generate(ctx context.Context, req MusicRequest) (*MusicResult, error) {
	if req.Prompt == "" {
		return nil, fmt.Errorf("fal music: empty prompt")
	}
	if req.Instrumental && req.Lyrics != "" {
		return nil, fmt.Errorf("fal music: lyrics and instrumental are exclusive")
	}
	endpoint := p.model
	if req.Model != "" {
		endpoint = req.Model
	}

	logger.Debug("fal music generate", zap.String("endpoint", endpoint),
		zap.Bool("instrumental", req.Instrumental), zap.Int("lyrics_len", len(req.Lyrics)))

	start := time.Now()
	raw, err := p.client.Run(ctx, endpoint, falMusicInput(endpoint, req))
	if err != nil {
		logger.Error("fal music generate failed", zap.String("endpoint", endpoint), zap.Error(err))
		return nil, fmt.Errorf("fal music generate: %w", err)
	}
	var out falMusicOutput
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("fal music generate: decode output: %w", err)
	}
	if out.Audio.URL == "" {
		return nil, fmt.Errorf("fal music generate: no audio url in output")
	}

	res := &MusicResult{
		URL:         out.Audio.URL,
		ContentType: out.Audio.ContentType,
		FileName:    out.Audio.FileName,
		FileSize:    out.Audio.FileSize,
		Duration:    out.Audio.Duration,
		Lyrics:      cleanLyrics(out.Lyrics),
		Model:       endpoint,
		CostUSD:     EstimateMusicCost(endpoint),
		Elapsed:     time.Since(start),
	}
	if res.ContentType == "" {
		res.ContentType = "audio/mpeg"
	}

	logger.Debug("fal music generated", zap.String("endpoint", endpoint),
		zap.Int64("bytes", res.FileSize), zap.Duration("elapsed", res.Elapsed), zap.Float64("cost_usd", res.CostUSD))

	if p.meter != nil {
		p.meter(UsageEvent{
			DebugSpanID:      DebugSpanIDFromCtx(ctx),
			CallerID:         MeterCallerIDFromCtx(ctx),
			Provider:         "fal",
			Model:            endpoint,
			Operation:        MeterOperationFromCtx(ctx),
			EstimatedCostUSD: res.CostUSD,
			Metadata: mergeMeterMetadata(ctx, map[string]any{
				"type":         "music_gen",
				"instrumental": req.Instrumental,
				"lyrics":       req.Lyrics != "",
				"bytes":        res.FileSize,
				"elapsed_ms":   res.Elapsed.Milliseconds(),
			}),
		})
	}
	return res, nil
}
