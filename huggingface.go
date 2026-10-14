package ai

import "github.com/sashabaranov/go-openai"

// DefaultHuggingFaceBaseURL is the HF Inference router's OpenAI-compatible endpoint.
// Override via HF_BASE_URL when targeting a dedicated endpoint or a specific upstream
// provider route (e.g. https://router.huggingface.co/<provider>/v1).
const DefaultHuggingFaceBaseURL = "https://router.huggingface.co/v1"

// HuggingFaceProvider speaks the HF Inference Router, which exposes an
// OpenAI-compatible Chat Completions API. Models are addressed by their
// canonical HF id (e.g. "meta-llama/Llama-3.3-70B-Instruct").
type HuggingFaceProvider struct {
	openAICompat
}

func (p *HuggingFaceProvider) WithModeration(m ModerationProvider) *HuggingFaceProvider {
	p.moderation = m
	return p
}

func (p *HuggingFaceProvider) WithMeter(hook MeterHook) *HuggingFaceProvider {
	p.meter = hook
	return p
}

// NewHuggingFaceProvider builds a provider against the HF router.
// baseURL may be empty to use DefaultHuggingFaceBaseURL.
func NewHuggingFaceProvider(apiKey, model, baseURL string) *HuggingFaceProvider {
	if baseURL == "" {
		baseURL = DefaultHuggingFaceBaseURL
	}
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = baseURL
	return &HuggingFaceProvider{newOpenAICompat(cfg, model, "huggingface", "HuggingFace")}
}
