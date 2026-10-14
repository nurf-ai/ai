package ai

import (
	"net/http"

	"github.com/sashabaranov/go-openai"
)

// DefaultOpenRouterBaseURL is OpenRouter's OpenAI-compatible endpoint.
const DefaultOpenRouterBaseURL = "https://openrouter.ai/api/v1"

// OpenRouterProvider speaks OpenRouter's OpenAI-compatible Chat Completions
// API. Models are addressed by OpenRouter id (e.g. "moonshotai/kimi-k3").
//
// OpenRouter prices a request by the upstream it routes to and reports the
// charge in usage.cost; that reported cost is what gets metered, so its
// models carry "pricing": "dynamic" in models.json instead of rates.
type OpenRouterProvider struct {
	openAICompat
}

func (p *OpenRouterProvider) WithModeration(m ModerationProvider) *OpenRouterProvider {
	p.moderation = m
	return p
}

func (p *OpenRouterProvider) WithMeter(hook MeterHook) *OpenRouterProvider {
	p.meter = hook
	return p
}

// NewOpenRouterProvider builds a provider against OpenRouter.
func NewOpenRouterProvider(apiKey, model string) *OpenRouterProvider {
	return newOpenRouterProvider(apiKey, model, DefaultOpenRouterBaseURL)
}

func newOpenRouterProvider(apiKey, model, baseURL string) *OpenRouterProvider {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	cfg.HTTPClient = &http.Client{Transport: reportedCostTransport{}}
	return &OpenRouterProvider{newOpenAICompat(cfg, model, "openrouter", "OpenRouter")}
}
