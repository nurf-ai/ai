package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

// Some gpt-5.x variants refuse function tools while reasoning is on
// (/v1/chat/completions, 400 "… set reasoning_effort to 'none'"). The provider
// must retry once with reasoning_effort "none" and remember it.
func TestOpenAIChat_ToolsRetryWithoutReasoning(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		_, hasTools := req["tools"]
		if hasTools && req["reasoning_effort"] != "none" {
			w.WriteHeader(400)
			_, _ = w.Write([]byte(`{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.","type":"invalid_request_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","model":"gpt-5.6-luna","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"say","arguments":"{\"text\":\"hi\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	p := NewOpenAIProviderWithBaseURL("k", "gpt-5.6-luna", srv.URL)
	tools := []Tool{{Name: "say", Description: "speak", Parameters: map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}}}}
	resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hello"}}, tools)
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "say" {
		t.Fatalf("resp = %+v", resp)
	}
	if calls.Load() != 2 {
		t.Fatalf("expected one 400 + one retry, got %d calls", calls.Load())
	}
	// sticky: the next tool call goes out right with "none"
	if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "again"}}, tools); err != nil {
		t.Fatalf("second chat: %v", err)
	}
	if calls.Load() != 3 {
		t.Fatalf("expected the retry to stick (3 calls total), got %d", calls.Load())
	}
}

func TestToolsRejectReasoning(t *testing.T) {
	msg := &openai.APIError{HTTPStatusCode: 400, Message: "Function tools with reasoning_effort are not supported for m in /v1/chat/completions"}
	if !toolsRejectReasoning(msg, true, "") || !toolsRejectReasoning(msg, true, "low") {
		t.Fatal("should retry with tools and reasoning on")
	}
	if toolsRejectReasoning(msg, false, "") {
		t.Fatal("no tools: nothing to retry")
	}
	if toolsRejectReasoning(msg, true, "none") {
		t.Fatal("already none: retrying would loop")
	}
	if toolsRejectReasoning(&openai.APIError{HTTPStatusCode: 400, Message: "invalid schema"}, true, "") {
		t.Fatal("unrelated 400")
	}
	if toolsRejectReasoning(&openai.APIError{HTTPStatusCode: 429, Message: "reasoning_effort tools"}, true, "") {
		t.Fatal("not a 400")
	}
	if toolsRejectReasoning(io.EOF, true, "") || toolsRejectReasoning(nil, true, "") {
		t.Fatal("non-API / nil error")
	}
}
