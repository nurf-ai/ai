package ai

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/genai"
)

// perM prices tokens at a per-million rate.
func perM(tokens int, rate float64) float64 { return float64(tokens) * rate / 1_000_000 }

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

// withTestPricing adds a models.json entry for the length of the test.
func withTestPricing(t *testing.T, model string, p modelPricing) {
	t.Helper()
	pricingTable[model] = p
	t.Cleanup(func() { delete(pricingTable, model) })
}

func TestEstimateCostCacheInPrompt(t *testing.T) {
	tests := []struct {
		name                     string
		model                    string
		prompt, out, write, read int
		want                     float64
	}{
		{"no cache", "gpt-5.4-mini", 10_000, 500, 0, 0,
			perM(10_000, 0.75) + perM(500, 4.5)},
		{"reads priced once at the cache-read rate", "gpt-5.4-mini", 10_000, 500, 0, 8_000,
			perM(2_000, 0.75) + perM(8_000, 0.075) + perM(500, 4.5)},
		{"reads and writes each priced once", "gpt-5.6-luna", 12_000, 100, 7_000, 4_000,
			perM(1_000, 0.2) + perM(7_000, 0.25) + perM(4_000, 0.02) + perM(100, 1.2)},
		{"writes without a cache-write rate stay ordinary input", "gpt-5.4-mini", 12_000, 0, 7_000, 4_000,
			perM(8_000, 0.75) + perM(4_000, 0.075)},
		{"reads without a cache-read rate stay ordinary input", "gemini-3.8-flash", 10_000, 500, 0, 8_000,
			perM(10_000, 0.75) + perM(500, 3.75)},
		{"cached above the prompt: ordinary part clamped at 0", "gpt-5.4-mini", 1_000, 0, 0, 3_000,
			perM(3_000, 0.075)},
		{"negative counts read as 0", "gpt-5.4-mini", -10, -10, -10, -10, 0},
		{"unknown model", "no-such-model", 10_000, 500, 0, 8_000, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := estimateCostCacheInPrompt(tt.model, tt.prompt, tt.out, tt.write, tt.read)
			if got < 0 || !near(got, tt.want) {
				t.Errorf("cost = %v, want %v", got, tt.want)
			}
		})
	}
}

// cachedCall is one call's usage as a provider that counts cached tokens
// inside the prompt total reports it.
type cachedCall struct{ prompt, cached, out int }

// TestCacheInPromptPricing_Providers meters fake usage through each provider
// that reports cached tokens as a subset of the prompt: the cached share is
// priced once, at the cache-read rate, never also at the input rate.
func TestCacheInPromptPricing_Providers(t *testing.T) {
	ctx := context.Background()
	withTestPricing(t, "test-gemini-cached", modelPricing{InputPerMillion: 0.5, OutputPerMillion: 3, CacheReadPerMillion: 0.05})
	providers := []struct {
		name  string
		model string
		emit  func(*meterTap, string, cachedCall)
	}{
		{"openai", "gpt-5.4-mini", func(tap *meterTap, model string, c cachedCall) {
			p := &OpenAIProvider{model: model, meter: tap.hook}
			p.emitUsage(ctx, openai.Usage{
				PromptTokens:        c.prompt,
				CompletionTokens:    c.out,
				TotalTokens:         c.prompt + c.out,
				PromptTokensDetails: &openai.PromptTokensDetails{CachedTokens: c.cached},
			}, "", "")
		}},
		{"gemini", "test-gemini-cached", func(tap *meterTap, model string, c cachedCall) {
			p := &GeminiProvider{model: model, meter: tap.hook}
			p.emitUsage(ctx, &genai.GenerateContentResponse{UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        int32(c.prompt),
				CandidatesTokenCount:    int32(c.out),
				CachedContentTokenCount: int32(c.cached),
				TotalTokenCount:         int32(c.prompt + c.out),
			}}, "", "")
		}},
	}
	for _, pv := range providers {
		t.Run(pv.name, func(t *testing.T) {
			rates := pricingTable[pv.model]
			meter := func(c cachedCall) UsageEvent {
				tap := &meterTap{}
				pv.emit(tap, pv.model, c)
				return tap.one(t)
			}
			uncached := meter(cachedCall{prompt: 10_000, out: 500})
			cached := meter(cachedCall{prompt: 10_000, cached: 8_000, out: 500})

			if want := EstimateCostFull(pv.model, 10_000, 500, 0, 0); !near(uncached.EstimatedCostUSD, want) {
				t.Errorf("uncached cost = %v, want %v (unchanged)", uncached.EstimatedCostUSD, want)
			}
			if cached.EstimatedCostUSD >= uncached.EstimatedCostUSD {
				t.Errorf("cached cost %v >= uncached %v", cached.EstimatedCostUSD, uncached.EstimatedCostUSD)
			}
			// the cached share moves from the input rate to the cache-read rate
			want := uncached.EstimatedCostUSD - perM(8_000, rates.InputPerMillion) + perM(8_000, rates.CacheReadPerMillion)
			if !near(cached.EstimatedCostUSD, want) {
				t.Errorf("cached cost = %v, want %v (cached share priced once)", cached.EstimatedCostUSD, want)
			}
			if cached.InputTokens != 10_000 || cached.CacheReadInputTokens != 8_000 || cached.TotalTokens != 10_500 {
				t.Errorf("event tokens = in %d, cache read %d, total %d; want the reported 10000, 8000, 10500",
					cached.InputTokens, cached.CacheReadInputTokens, cached.TotalTokens)
			}

			over := meter(cachedCall{prompt: 1_000, cached: 3_000})
			if want := perM(3_000, rates.CacheReadPerMillion); over.EstimatedCostUSD < 0 || !near(over.EstimatedCostUSD, want) {
				t.Errorf("cached > prompt: cost = %v, want %v", over.EstimatedCostUSD, want)
			}
		})
	}
}

// A Gemini model with no cache-read rate prices its cached share as ordinary
// input, once: not free, not twice.
func TestGeminiCachedTokens_NoCacheReadRate(t *testing.T) {
	const model = "gemini-3.8-flash"
	if pricingTable[model].CacheReadPerMillion != 0 {
		t.Skipf("%s has a cache-read rate now; covered by TestCacheInPromptPricing_Providers", model)
	}
	tap := &meterTap{}
	p := &GeminiProvider{model: model, meter: tap.hook}
	p.emitUsage(context.Background(), &genai.GenerateContentResponse{UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount: 10_000, CandidatesTokenCount: 500, CachedContentTokenCount: 8_000,
	}}, "", "")
	if got, want := tap.one(t).EstimatedCostUSD, EstimateCostFull(model, 10_000, 500, 0, 0); !near(got, want) {
		t.Errorf("cost = %v, want %v", got, want)
	}
}

// Anthropic reports cache reads and writes on top of input_tokens: its
// pricing adds them, unchanged.
func TestAnthropicCachePricing_Unchanged(t *testing.T) {
	const model = "claude-haiku-4-5" // in $1, out $5, cache write $1.25, cache read $0.10
	tests := []struct {
		name            string
		in, out, cw, cr int
		want            float64
	}{
		{"no cache", 1_000, 100, 0, 0, perM(1_000, 1) + perM(100, 5)},
		{"cache read on top of input", 5_000, 1_000, 0, 30_000, perM(5_000, 1) + perM(1_000, 5) + perM(30_000, 0.1)},
		{"cache write on top of input", 1_000, 100, 2_000, 0, perM(1_000, 1) + perM(100, 5) + perM(2_000, 1.25)},
		{"read and write", 500, 200, 2_000, 30_000, perM(500, 1) + perM(200, 5) + perM(2_000, 1.25) + perM(30_000, 0.1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tap := &meterTap{}
			p := &AnthropicProvider{model: model, meter: tap.hook}
			p.emitUsage(context.Background(), &anthropic.Message{Usage: anthropic.Usage{
				InputTokens:              int64(tt.in),
				OutputTokens:             int64(tt.out),
				CacheCreationInputTokens: int64(tt.cw),
				CacheReadInputTokens:     int64(tt.cr),
			}}, "", "")
			ev := tap.one(t)
			if !near(ev.EstimatedCostUSD, tt.want) {
				t.Errorf("cost = %v, want %v", ev.EstimatedCostUSD, tt.want)
			}
			if ev.InputTokens != tt.in || ev.CacheCreationInputTokens != tt.cw || ev.CacheReadInputTokens != tt.cr {
				t.Errorf("event tokens = %+v", ev)
			}
		})
	}
}

// OpenAI reports cache writes in usage.prompt_tokens_details.cache_write_tokens,
// part of prompt_tokens like the cached reads. The OpenAI client's Usage type
// drops the field; the provider reads it off the body and prices the writes
// at the cache-write rate, once.
func TestOpenAI_CacheWritesPricedOnce(t *testing.T) {
	const model = "gpt-5.6-luna" // in $0.20, out $1.20, cache write $0.25, cache read $0.02
	const usage = `"usage":{"prompt_tokens":12000,"completion_tokens":100,"total_tokens":12100,"prompt_tokens_details":{"cached_tokens":4000,"cache_write_tokens":7000}}`
	want := perM(1_000, 0.2) + perM(7_000, 0.25) + perM(4_000, 0.02) + perM(100, 1.2)
	serve := func(t *testing.T, contentType, body string) string {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", contentType)
			_, _ = fmt.Fprint(w, body)
		}))
		t.Cleanup(srv.Close)
		return srv.URL
	}
	check := func(t *testing.T, ev UsageEvent) {
		t.Helper()
		if ev.InputTokens != 12_000 || ev.CacheReadInputTokens != 4_000 || ev.CacheCreationInputTokens != 7_000 {
			t.Errorf("event tokens = in %d, cache read %d, cache write %d; want 12000, 4000, 7000",
				ev.InputTokens, ev.CacheReadInputTokens, ev.CacheCreationInputTokens)
		}
		if !near(ev.EstimatedCostUSD, want) {
			t.Errorf("cost = %v, want %v", ev.EstimatedCostUSD, want)
		}
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	t.Run("chat", func(t *testing.T) {
		url := serve(t, "application/json", `{"id":"x","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],`+usage+`}`)
		tap := &meterTap{}
		p := NewOpenAIProviderWithBaseURL("k", model, url).WithMeter(tap.hook)
		ctx := context.Background()
		for range 2 { // each call reads its own writes, none carried over
			if _, err := p.Chat(ctx, msgs, nil); err != nil {
				t.Fatal(err)
			}
		}
		if len(tap.events) != 2 {
			t.Fatalf("events = %d, want 2", len(tap.events))
		}
		for _, ev := range tap.events {
			check(t, ev)
		}
	})

	t.Run("stream", func(t *testing.T) {
		body := `data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}` + "\n\n" +
			`data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			`data: {"id":"x","object":"chat.completion.chunk","choices":[],` + usage + `}` + "\n\n" +
			"data: [DONE]\n\n"
		tap := &meterTap{}
		p := NewOpenAIProviderWithBaseURL("k", model, serve(t, "text/event-stream", body)).WithMeter(tap.hook)
		if _, err := p.ChatStream(context.Background(), msgs, nil, func(StreamChunk) error { return nil }); err != nil {
			t.Fatal(err)
		}
		check(t, tap.one(t))
	})

	t.Run("structured", func(t *testing.T) {
		url := serve(t, "application/json", `{"id":"x","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"{\"message\":\"hi\"}"}}],`+usage+`}`)
		tap := &meterTap{}
		p := NewOpenAIProviderWithBaseURL("k", model, url).WithMeter(tap.hook)
		var out meterGreeting
		if err := p.CreateStructuredOutput(context.Background(), "greet me", "greet", &out); err != nil {
			t.Fatal(err)
		}
		check(t, tap.one(t))
	})
}
