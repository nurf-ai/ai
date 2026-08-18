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
