package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Chat and ChatStream send WithMaxTokens as max_completion_tokens, 4096 when
// unset: a reply longer than 4096 tokens used to be cut there whatever the
// caller asked for.
func TestOpenAIChat_HonorsMaxTokens(t *testing.T) {
	var got []float64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		n, _ := req["max_completion_tokens"].(float64)
		got = append(got, n)
		if req["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	p := NewOpenAIProviderWithBaseURL("k", "gpt-4.1-mini", srv.URL)
	msgs := []Message{{Role: RoleUser, Content: "hi"}}
	if _, err := p.Chat(WithMaxTokens(context.Background(), 8192), msgs, nil); err != nil {
		t.Fatalf("chat: %v", err)
	}
	if _, err := p.Chat(context.Background(), msgs, nil); err != nil {
		t.Fatalf("chat: %v", err)
	}
	if _, err := p.ChatStream(WithMaxTokens(context.Background(), 12288), msgs, nil, func(StreamChunk) error { return nil }); err != nil {
		t.Fatalf("chat stream: %v", err)
	}
	want := []float64{8192, 4096, 12288}
	if len(got) != len(want) {
		t.Fatalf("requests = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("request %d: max_completion_tokens = %v, want %v", i, got[i], want[i])
		}
	}
}
