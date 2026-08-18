package ai

import (
	"fmt"
	"strings"
	"sync"

	tiktoken "github.com/pkoukk/tiktoken-go"
	tiktokenloader "github.com/pkoukk/tiktoken-go-loader"
)

// modelMaxInputTokens maps "<company>/<model>" to the model's advertised
// maximum input context window in tokens. Populated from models.json at init.
var modelMaxInputTokens map[string]int64

// stripDateSuffix removes a trailing "-YYYYMMDD" tag from a model id.
// Anthropic ships model ids like "claude-sonnet-4-5-20250514"; callers
// pass the dated form but the limits table keeps only bare names.
// Returns the input unchanged if no date suffix is detected.
func stripDateSuffix(model string) string {
	if len(model) < 9 || model[len(model)-9] != '-' {
		return model
	}
	tail := model[len(model)-8:]
	for i := range 8 {
		if tail[i] < '0' || tail[i] > '9' {
			return model
		}
	}
	return model[:len(model)-9]
}

// MaxInputTokensLLM returns the advertised maximum input context window
// in tokens for the given model.
//
// company is the model-family owner / API namespace — one of "anthropic",
// "openai", or "huggingface". modelName is the exact model id as passed
// to the provider constructor (for HF that's the full "<org>/<model>" id).
// Trailing date suffixes like "-20250514" are stripped automatically.
//
// Returns an error if the model is unknown. Callers that need a safe
// default should handle the error explicitly rather than rely on a
// fallback — silently returning 0 or a guess would mask config typos.
func MaxInputTokensLLM(company, modelName string) (int64, error) {
	// Exact match first (covers any explicitly-dated entry if we ever add one).
	if v, ok := modelMaxInputTokens[company+"/"+modelName]; ok {
		return v, nil
	}
	// Fall back to the stripped form.
	if stripped := stripDateSuffix(modelName); stripped != modelName {
		if v, ok := modelMaxInputTokens[company+"/"+stripped]; ok {
			return v, nil
		}
	}
	return 0, fmt.Errorf("ai: unknown model %q for company %q", modelName, company)
}

// --- Token counting ---------------------------------------------------

// Token counting uses tiktoken's o200k_base BPE — OpenAI's newest general
// encoding (used by GPT-4o / GPT-5). It's exact for OpenAI models and a
// reasonable approximation (within ~15%) for Anthropic / HF model families.
// Anthropic and HF vendors ship their own tokenizers, but loading their
// vocab at runtime isn't worth the binary weight for "am I near the limit"
// budget checks. If exact counts matter for a given provider, swap in a
// model-specific path later.
//
// We use tiktoken-go-loader so the BPE vocab ships embedded — no runtime
// download on first call.
var (
	tokenEncoderOnce sync.Once
	tokenEncoder     *tiktoken.Tiktoken
	tokenEncoderErr  error
)

func getTokenEncoder() (*tiktoken.Tiktoken, error) {
	tokenEncoderOnce.Do(func() {
		tiktoken.SetBpeLoader(tiktokenloader.NewOfflineLoader())
		enc, err := tiktoken.GetEncoding("o200k_base")
		if err != nil {
			tokenEncoderErr = fmt.Errorf("ai: load o200k_base encoder: %w", err)
			return
		}
		tokenEncoder = enc
	})
	return tokenEncoder, tokenEncoderErr
}

// CountTokens returns an approximate token count for s using the
// o200k_base BPE encoding. Exact for GPT-4o / GPT-5 family; within ~15%
// for Anthropic and HF model families. Empty string returns 0.
//
// Falls back to a len(s)/4 heuristic if the tiktoken encoder fails to
// initialize, so this never returns an error — a usable rough number is
// more helpful at call sites than a plumbed error path.
func CountTokens(s string) int {
	if s == "" {
		return 0
	}
	enc, err := getTokenEncoder()
	if err != nil || enc == nil {
		return fallbackTokenEstimate(s)
	}
	return len(enc.Encode(s, nil, nil))
}

// CountTokensOver reports whether counting tokens in s exceeds limit.
// Short-circuits using the cheap char-based upper bound before invoking
// the BPE encoder, so callers can gate long strings without paying the
// full tokenization cost when they're obviously under the limit.
func CountTokensOver(s string, limit int64) bool {
	if limit <= 0 {
		return true
	}
	// A token is at least ~1 char in BPE, so len(s) > limit is a hard fail
	// and len(s)/4 < limit * 0.5 is a safe "definitely under" shortcut.
	if int64(len(s)) <= limit/2 {
		return false
	}
	return int64(CountTokens(s)) > limit
}

func fallbackTokenEstimate(s string) int {
	// Same heuristic queue/session.go used before this utility existed.
	n := len(strings.TrimSpace(s)) / 4
	if n == 0 && s != "" {
		return 1
	}
	return n
}
