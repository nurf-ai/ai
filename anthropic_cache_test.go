package ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// Check the actual wire bytes: equivalent JSON with reordered properties can
// invalidate the tools prefix, including the system prompt cached after it.
func TestAnthropicToolSchemaCacheStability(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"},"days":{"type":"integer"},"options":{"type":"object","properties":{"units":{"type":"string"},"language":{"type":"string"}}},"region":{"type":"string"},"timezone":{"type":"string"}},"required":["city"]}`)
	var parameters map[string]any
	if err := json.Unmarshal(schema, &parameters); err != nil {
		t.Fatal(err)
	}
	tools := []Tool{{Name: "get_weather", Description: "Get the weather", Parameters: parameters}}

	for _, mode := range []string{"chat", "stream", "structured"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			type request struct {
				Tools  json.RawMessage `json:"tools"`
				System []struct {
					Text         string        `json:"text"`
					CacheControl *CacheControl `json:"cache_control"`
				} `json:"system"`
			}
			requests := make(chan request, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req request
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode request: %v", err)
					http.Error(w, "invalid request", http.StatusBadRequest)
					return
				}
				requests <- req
				if mode == "stream" {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = w.Write([]byte(anthropicStream("end_turn", true)))
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"tool_use","id":"toolu_1","name":"structured_output","input":{"city":"Paris"}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":1}}`))
			}))
			t.Cleanup(srv.Close)
			p := anthropicTestProvider(srv)
			ctx := WithCacheSysPrompt(t.Context())
			var first json.RawMessage
			for i := range 30 {
				live := fmt.Sprintf("Current request: %d", i)
				messages := []Message{
					{Role: RoleSystem, Content: "Stable instructions", CacheControl: &CacheControl{Type: "ephemeral"}},
					{Role: RoleSystem, Content: live},
					{Role: RoleUser, Content: "What is the weather?"},
				}
				var err error
				switch mode {
				case "chat":
					_, err = p.Chat(ctx, messages, tools)
				case "stream":
					_, err = p.ChatStream(ctx, messages, tools, func(StreamChunk) error { return nil })
				case "structured":
					_, err = p.CreateStructuredOutputFromSchema(ctx, live, "Stable instructions", schema)
				}
				if err != nil {
					t.Fatalf("request %d: %v", i, err)
				}
				var req request
				select {
				case req = <-requests:
				default:
					t.Fatal("provider did not send a request")
				}
				if i == 0 {
					first = req.Tools
					var sent []struct {
						InputSchema map[string]any `json:"input_schema"`
					}
					if err := json.Unmarshal(first, &sent); err != nil {
						t.Fatal(err)
					}
					if len(sent) != 1 {
						t.Fatalf("got %d tools, want 1", len(sent))
					}
					if diff := cmp.Diff(parameters, sent[0].InputSchema); diff != "" {
						t.Fatalf("tool schema changed (-want +got):\n%s", diff)
					}
				} else if string(req.Tools) != string(first) {
					t.Fatalf("request %d changed the tool prefix:\nfirst: %s\nnow:   %s", i, first, req.Tools)
				}
				if len(req.System) == 0 || req.System[0].Text != "Stable instructions" || req.System[0].CacheControl == nil || req.System[0].CacheControl.Type != "ephemeral" {
					t.Fatalf("request %d lost the stable system cache boundary: %+v", i, req.System)
				}
				if mode != "structured" && (len(req.System) != 2 || req.System[1].Text != live) {
					t.Fatalf("request %d lost live context: %+v", i, req.System)
				}
			}
		})
	}
}
