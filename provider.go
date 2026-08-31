package ai

import (
	"context"
	"fmt"
	"os"

	"go.uber.org/zap"
)

// hfBaseURL returns the HF router base URL, optionally overridden by env.
func hfBaseURL() string { return os.Getenv("HF_BASE_URL") }

// Role constants + Tool/Message/Response/Part/ToolCall/LLMProvider
// are defined in types.go + ai_contract.go.

// ProviderOption configures an LLMProvider created by NewLLMProvider.
type ProviderOption func(*providerConfig)

type providerConfig struct {
	meter      MeterHook
	moderation ModerationProvider
}

// WithMeterOption returns a ProviderOption that attaches a meter hook.
func WithMeterOption(hook MeterHook) ProviderOption {
	return func(c *providerConfig) { c.meter = hook }
}

// WithModerationOption returns a ProviderOption that attaches a moderation provider.
func WithModerationOption(m ModerationProvider) ProviderOption {
	return func(c *providerConfig) { c.moderation = m }
}

// NewLLMProvider creates an LLMProvider from a provider name
// ("openai", "anthropic", "gemini", "ollama", or "huggingface").
// For ollama, apiKey is the base URL (e.g. "http://ollama:11434/v1").
// For huggingface, apiKey is the HF token; base URL is read from HF_BASE_URL
// (defaults to DefaultHuggingFaceBaseURL).
// For gemini, apiKey is the Google AI API key.
func NewLLMProvider(providerName, apiKey, model string, opts ...ProviderOption) LLMProvider {
	var cfg providerConfig
	for _, o := range opts {
		o(&cfg)
	}
	switch providerName {
	case "openai":
		p := NewOpenAIProvider(apiKey, model)
		if cfg.moderation != nil {
			p.WithModeration(cfg.moderation)
		}
		if cfg.meter != nil {
			p.WithMeter(cfg.meter)
		}
		return p
	case "gemini":
		p, err := NewGeminiProvider(context.Background(), apiKey, model)
		if err != nil {
			logger.Error("failed to create gemini provider", zap.Error(err))
			return nil
		}
		if cfg.moderation != nil {
			p.WithModeration(cfg.moderation)
		}
		if cfg.meter != nil {
			p.WithMeter(cfg.meter)
		}
		return p
	case "ollama":
		p := NewOllamaProvider(apiKey, model)
		if cfg.moderation != nil {
			p.WithModeration(cfg.moderation)
		}
		if cfg.meter != nil {
			p.WithMeter(cfg.meter)
		}
		return p
	case "huggingface":
		p := NewHuggingFaceProvider(apiKey, model, hfBaseURL())
		if cfg.moderation != nil {
			p.WithModeration(cfg.moderation)
		}
		if cfg.meter != nil {
			p.WithMeter(cfg.meter)
		}
		return p
	default:
		p := NewAnthropicProvider(apiKey, model)
		if cfg.moderation != nil {
			p.WithModeration(cfg.moderation)
		}
		if cfg.meter != nil {
			p.WithMeter(cfg.meter)
		}
		return p
	}
}

// NewEmbedder creates an Embedder from a provider name + API key.
// Only OpenAI is supported for now — returns nil for other providers.
func NewEmbedder(providerName, apiKey string) Embedder {
	switch providerName {
	case "openai":
		return NewOpenAIProvider(apiKey, "")
	default:
		return nil
	}
}

// EmbedderFromLLM extracts an Embedder from an LLMProvider if it
// satisfies the Embedder interface (e.g. OpenAIProvider).
func EmbedderFromLLM(llm LLMProvider) Embedder {
	if e, ok := llm.(Embedder); ok {
		return e
	}
	return nil
}

// LLMModerable is implemented by providers that accept a moderation provider.
type LLMModerable interface {
	SetModeration(ModerationProvider)
}

// SetLLMModeration attaches a moderation provider to any LLMProvider that
// satisfies LLMModerable.
func SetLLMModeration(llm LLMProvider, m ModerationProvider) {
	if m == nil {
		return
	}
	if p, ok := llm.(LLMModerable); ok {
		p.SetModeration(m)
	}
}

// NewSTTProvider creates an STTProvider from a provider name + API key + model.
// Only OpenAI is supported for now — returns nil for other providers.
func NewSTTProvider(providerName, apiKey, model string) STTProvider {
	switch providerName {
	case "openai":
		return newOpenAISTTProvider(apiKey, model)
	default:
		return nil
	}
}

// NewImageProvider creates an ImageProvider from a provider name ("openai" or "gemini").
func NewImageProvider(ctx context.Context, providerName, apiKey, model string) (ImageProvider, error) {
	switch providerName {
	case "openai":
		return newOpenAIImageProvider(apiKey, model), nil
	case "gemini":
		return newGeminiImageProvider(ctx, apiKey, model)
	default:
		return nil, fmt.Errorf("unsupported image provider: %s", providerName)
	}
}

// NewVideoProvider creates a VideoProvider from a provider name ("fal").
// For fal, model is the full endpoint id (default DefaultFalVideoModel).
func NewVideoProvider(providerName, apiKey, model string) (VideoProvider, error) {
	switch providerName {
	case "fal":
		return newFalVideoProvider(apiKey, model), nil
	default:
		return nil, fmt.Errorf("unsupported video provider: %s", providerName)
	}
}

// SetImageModeration attaches a moderation provider to any ImageProvider
// that satisfies the moderable interface.
func SetImageModeration(img ImageProvider, m ModerationProvider) {
	if m == nil || img == nil {
		return
	}
	type moderable interface{ SetModeration(ModerationProvider) }
	if p, ok := img.(moderable); ok {
		p.SetModeration(m)
	}
}
