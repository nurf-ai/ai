package ai

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const openRouterCompletion = `{"id":"gen-1","object":"chat.completion","model":"moonshotai/kimi-k3",
"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
"usage":{"prompt_tokens":87,"completion_tokens":66,"total_tokens":153,"cost":0.0010283}}`

func openRouterServer(t *testing.T, body string, sse bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if sse {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func meterSpy() (MeterHook, *[]UsageEvent) {
	var evs []UsageEvent
	return func(ev UsageEvent) { evs = append(evs, ev) }, &evs
}

func TestOpenRouter_ChatMetersReportedCost(t *testing.T) {
	srv := openRouterServer(t, openRouterCompletion, false)
	hook, evs := meterSpy()
	p := newOpenRouterProvider("k", "moonshotai/kimi-k3", srv.URL).WithMeter(hook)
	resp, err := p.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hi" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(*evs) != 1 {
		t.Fatalf("meter events = %d, want 1", len(*evs))
	}
	ev := (*evs)[0]
	if ev.Provider != "openrouter" || ev.InputTokens != 87 || ev.OutputTokens != 66 {
		t.Errorf("event = %+v", ev)
	}
	if math.Abs(ev.EstimatedCostUSD-0.0010283) > 1e-12 {
		t.Errorf("cost = %v, want reported 0.0010283", ev.EstimatedCostUSD)
	}
}

func TestOpenRouter_StreamMetersReportedCost(t *testing.T) {
	chunks := []string{
		`{"id":"gen-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
		`{"id":"gen-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cost":0.00042}}`,
	}
	var body strings.Builder
	body.WriteString(": OPENROUTER PROCESSING\n\n")
	for _, c := range chunks {
		body.WriteString("data: " + c + "\n\n")
	}
	body.WriteString("data: [DONE]\n\n")
	srv := openRouterServer(t, body.String(), true)
	hook, evs := meterSpy()
	p := newOpenRouterProvider("k", "moonshotai/kimi-k3", srv.URL).WithMeter(hook)
	resp, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil, func(StreamChunk) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hi" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(*evs) != 1 {
		t.Fatalf("meter events = %d, want 1", len(*evs))
	}
	if got := (*evs)[0].EstimatedCostUSD; math.Abs(got-0.00042) > 1e-12 {
		t.Errorf("cost = %v, want reported 0.00042", got)
	}
}

// Without a reported cost a dynamic-priced model records 0 and does not
// trip the unknown-model hook: it is known, just not priced statically.
func TestOpenRouter_NoReportedCost(t *testing.T) {
	srv := openRouterServer(t, strings.Replace(openRouterCompletion, `,"cost":0.0010283`, "", 1), false)
	var unknown []string
	SetUnknownModelHook(func(m string) { unknown = append(unknown, m) })
	defer SetUnknownModelHook(nil)
	hook, evs := meterSpy()
	p := newOpenRouterProvider("k", "moonshotai/kimi-k3", srv.URL).WithMeter(hook)
	if _, err := p.Chat(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatal(err)
	}
	if len(*evs) != 1 || (*evs)[0].EstimatedCostUSD != 0 {
		t.Errorf("events = %+v, want one at cost 0", *evs)
	}
	if len(unknown) != 0 {
		t.Errorf("unknown-model hook fired for %v", unknown)
	}
}

func TestOpenRouter_MaxInputTokens(t *testing.T) {
	n, err := NewOpenRouterProvider("k", "moonshotai/kimi-k3").MaxInputTokens()
	if err != nil || n != 1048576 {
		t.Errorf("MaxInputTokens = %d, %v", n, err)
	}
}
