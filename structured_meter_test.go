package ai

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

// structuredReplies serves one chat completion per request, in order, each
// with usage {in: 10, out: 5} and cost.
func structuredReplies(t *testing.T, cost float64, contents ...string) *httptest.Server {
	t.Helper()
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		content := contents[min(n, len(contents)-1)]
		n++
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"x","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":%q}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"cost":%g}}`, content, cost)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// CreateStructuredOutput (instructor) used to drop the usage it got back,
// so structured calls were free in the meter.
func TestCreateStructuredOutput_Metered(t *testing.T) {
	srv := structuredReplies(t, 0, `{"message":"hi"}`)
	for name, p := range map[string]interface {
		LLMProvider
		SetMeter(MeterHook)
	}{
		"openai":      NewOpenAIProviderWithBaseURL("k", "gpt-4o-mini", srv.URL),
		"ollama":      NewOllamaProvider(srv.URL, "gpt-oss:20b"),
		"huggingface": NewHuggingFaceProvider("k", "moonshotai/Kimi-K3", srv.URL),
	} {
		t.Run(name, func(t *testing.T) {
			hook, evs := meterSpy()
			p.SetMeter(hook)
			var out meterGreeting
			if err := p.CreateStructuredOutput(context.Background(), "greet me", "greet", &out); err != nil {
				t.Fatal(err)
			}
			if out.Message != "hi" {
				t.Errorf("out = %+v", out)
			}
			if len(*evs) != 1 || (*evs)[0].InputTokens != 10 || (*evs)[0].OutputTokens != 5 {
				t.Errorf("events = %+v, want one with 10 in / 5 out", *evs)
			}
		})
	}
}

// A retry is a second billed request: tokens and reported cost both sum.
func TestOpenRouterStructuredOutput_RetrySums(t *testing.T) {
	srv := structuredReplies(t, 0.001, `not json`, `{"message":"hi"}`)
	hook, evs := meterSpy()
	p := newOpenRouterProvider("k", "moonshotai/kimi-k3", srv.URL).WithMeter(hook)
	var out meterGreeting
	if err := p.CreateStructuredOutput(context.Background(), "greet me", "greet", &out); err != nil {
		t.Fatal(err)
	}
	if len(*evs) != 1 {
		t.Fatalf("events = %d, want 1", len(*evs))
	}
	ev := (*evs)[0]
	if ev.InputTokens != 20 || ev.OutputTokens != 10 || math.Abs(ev.EstimatedCostUSD-0.002) > 1e-12 {
		t.Errorf("event = %d in, %d out, $%v; want 20, 10, $0.002", ev.InputTokens, ev.OutputTokens, ev.EstimatedCostUSD)
	}
}

// named: instructor derives its schema name from the type
type meterGreeting struct {
	Message string `json:"message"`
}
