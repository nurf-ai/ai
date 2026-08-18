package ai

import (
	"context"
	"encoding/json"
)

// LLMProvider is the consumer-facing contract for any LLM backend.
type LLMProvider interface {
	Name() string
	Model() string
	CreateStructuredOutput(ctx context.Context, userPrompt, sysPrompt string, structuredOutput any) error
	CreateStructuredOutputFromSchema(ctx context.Context, userPrompt, sysPrompt string, schema json.RawMessage) (map[string]any, error)
	Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error)
}

// GenerateTyped wraps CreateStructuredOutput with Go generics so callers
// get back a typed pointer without pre-allocating the output struct.
func GenerateTyped[T any](ctx context.Context, p LLMProvider, userPrompt, sysPrompt string) (*T, error) {
	var out T
	if err := p.CreateStructuredOutput(ctx, userPrompt, sysPrompt, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Embedder produces vector embeddings for text. Consumers that need
// semantic similarity (discovery, memory recall, search ranking) depend
// on this interface. Not all LLM providers support embeddings — wire a
// separate provider if the main LLM doesn't (e.g. Anthropic + OpenAI
// embedding sidecar).
type Embedder interface {
	EmbedText(ctx context.Context, texts []string) ([][]float32, error)
	EmbedDimensions() int
}

type ctxKeyMaxTokens struct{}

func WithMaxTokens(ctx context.Context, n int) context.Context {
	return context.WithValue(ctx, ctxKeyMaxTokens{}, n)
}

func MaxTokensFromCtx(ctx context.Context, fallback int) int {
	if v, ok := ctx.Value(ctxKeyMaxTokens{}).(int); ok && v > 0 {
		return v
	}
	return fallback
}

type ctxKeyReasoningEffort struct{}

// WithReasoningEffort sets the reasoning effort level ("low", "medium", "high")
// for providers that support it.
// OpenAI: maps to reasoning_effort in chat completions.
// Anthropic: maps to extended thinking with a budget derived from effort level.
func WithReasoningEffort(ctx context.Context, effort string) context.Context {
	return context.WithValue(ctx, ctxKeyReasoningEffort{}, effort)
}

func ReasoningEffortFromCtx(ctx context.Context) string {
	if v, ok := ctx.Value(ctxKeyReasoningEffort{}).(string); ok {
		return v
	}
	return ""
}

// ModelFunc is the signature of an LLM Chat call, abstracted from any
// concrete provider.
type ModelFunc func(ctx context.Context, msgs []Message, tools []Tool) (*Response, error)

// Middleware wraps a ModelFunc with cross-cutting behaviour.
type Middleware func(next ModelFunc) ModelFunc

// Chain composes middleware so the first in the list is outermost.
func Chain(mws ...Middleware) Middleware {
	return func(final ModelFunc) ModelFunc {
		for i := len(mws) - 1; i >= 0; i-- {
			final = mws[i](final)
		}
		return final
	}
}
