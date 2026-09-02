package ai

import (
	_ "embed"
	"encoding/json"
	"maps"
	"sort"
	"strings"

	"go.uber.org/zap"
)

var unknownModelHook func(model string)

func SetUnknownModelHook(hook func(model string)) { unknownModelHook = hook }

func PricingTable() map[string]modelPricing {
	out := make(map[string]modelPricing, len(pricingTable))
	maps.Copy(out, pricingTable)
	return out
}

type modelPricing struct {
	InputPerMillion         float64 `json:"input_per_million"`
	OutputPerMillion        float64 `json:"output_per_million"`
	CacheCreationPerMillion float64 `json:"cache_creation_per_million,omitempty"`
	CacheReadPerMillion     float64 `json:"cache_read_per_million,omitempty"`
	FlatPerImage            float64 `json:"flat_per_image,omitempty"`
	MaxInputTokens          int64   `json:"max_input_tokens,omitempty"`
	PerVideoSecond             float64            `json:"per_video_second,omitempty"`
	PerVideoSecondByResolution map[string]float64 `json:"per_video_second_by_resolution,omitempty"`
	ImageOutputPerMillion      float64            `json:"image_output_per_million,omitempty"`
	VideoOutputPerMillion      float64            `json:"video_output_per_million,omitempty"`
}

// IsVideoModel reports whether model is priced per second of video.
func IsVideoModel(model string) bool {
	p, ok := pricingTable[model]
	return ok && p.PerVideoSecond > 0
}

// VideoModels lists every priced video model id, sorted.
func VideoModels() []string {
	var out []string
	for m, p := range pricingTable {
		if p.PerVideoSecond > 0 {
			out = append(out, m)
		}
	}
	sort.Strings(out)
	return out
}

//go:embed models.json
var pricingJSON []byte

var pricingTable map[string]modelPricing

func init() {
	var grouped map[string]map[string]modelPricing
	if err := json.Unmarshal(pricingJSON, &grouped); err != nil {
		panic("ai: parse models.json: " + err.Error())
	}
	pricingTable = make(map[string]modelPricing)
	modelMaxInputTokens = make(map[string]int64)
	for company, models := range grouped {
		for model, pricing := range models {
			pricingTable[model] = pricing
			if pricing.MaxInputTokens > 0 {
				modelMaxInputTokens[company+"/"+model] = pricing.MaxInputTokens
			}
		}
	}
}

func EstimateCostFull(model string, in, out, cw, cr int) float64 {
	p, ok := pricingTable[model]
	if !ok {
		logger.Warn("unknown model — cost recorded as 0",
			zap.String("model", model), zap.Int("in", in), zap.Int("out", out), zap.Int("cw", cw), zap.Int("cr", cr))
		if unknownModelHook != nil {
			unknownModelHook(model)
		}
		return 0
	}
	if p.FlatPerImage > 0 {
		return p.FlatPerImage
	}
	return (float64(in)*p.InputPerMillion +
		float64(out)*p.OutputPerMillion +
		float64(cw)*p.CacheCreationPerMillion +
		float64(cr)*p.CacheReadPerMillion) / 1_000_000
}

// EstimateVideoCost prices seconds of generated video for a per-second model.
// resolution selects a per-resolution rate when the table has one; otherwise
// the base rate applies. Unknown models are recorded as 0 (and reported via
// the unknown-model hook), like EstimateCostFull.
func EstimateVideoCost(model string, seconds float64, resolution string) float64 {
	p, ok := pricingTable[model]
	if !ok {
		logger.Warn("unknown video model — cost recorded as 0", zap.String("model", model), zap.Float64("seconds", seconds))
		if unknownModelHook != nil {
			unknownModelHook(model)
		}
		return 0
	}
	rate := p.PerVideoSecond
	if r, ok := p.PerVideoSecondByResolution[resolution]; ok && r > 0 {
		rate = r
	}
	if rate <= 0 || seconds <= 0 {
		return 0
	}
	return rate * seconds
}

// EstimateVideoCostByTokens prices a video generation from actual token
// counts returned by the provider. Falls back to 0 when the model has no
// per-token video pricing.
func EstimateVideoCostByTokens(model string, inputTokens, videoOutputTokens int) float64 {
	p, ok := pricingTable[model]
	if !ok || p.VideoOutputPerMillion <= 0 {
		return 0
	}
	cost := float64(videoOutputTokens) * p.VideoOutputPerMillion / 1_000_000
	if p.InputPerMillion > 0 {
		cost += float64(inputTokens) * p.InputPerMillion / 1_000_000
	}
	return cost
}

// EstimateImageCostByTokens prices an image generation from output tokens.
// Falls back to FlatPerImage when the model has no per-token image pricing.
func EstimateImageCostByTokens(model string, outputTokens int) float64 {
	p, ok := pricingTable[model]
	if !ok {
		return 0
	}
	if p.ImageOutputPerMillion > 0 && outputTokens > 0 {
		return float64(outputTokens) * p.ImageOutputPerMillion / 1_000_000
	}
	return p.FlatPerImage
}

var deprecatedModels = map[string]bool{
	"claude-3-5-sonnet": true,
	"claude-3-opus":     true,
	"claude-opus-4":     true,
	"claude-opus-4-1":   true,
	"claude-sonnet-4":   true,
}

func ModelsForProvider(provider string) []string {
	prefix := provider + "/"
	var models []string
	for key := range modelMaxInputTokens {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		model := key[len(prefix):]
		if stripDateSuffix(model) != model {
			continue
		}
		if deprecatedModels[model] {
			continue
		}
		if p, ok := pricingTable[model]; ok && p.FlatPerImage > 0 {
			continue
		}
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}
