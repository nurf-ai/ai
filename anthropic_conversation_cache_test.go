package ai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAnthropicConversationCacheWire(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, kind := range []string{"text", "image", "tool_use", "tool_result"} {
			t.Run(kind+map[bool]string{true: "/stream", false: "/chat"}[stream], func(t *testing.T) {
				marked := Message{Role: RoleUser, Content: "hello", CacheControl: &CacheControl{Type: "ephemeral"}}
				switch kind {
				case "image":
					marked.Parts = []Part{TextPart{Text: "look"}, ImagePart{MediaType: "image/png", Data: "aGVsbG8="}}
				case "tool_use":
					marked.Role = RoleAssistant
					marked.Content = ""
					marked.ToolCalls = []ToolCall{{ID: "call_1", Name: "get_weather", Arguments: json.RawMessage(`{}`)}}
				case "tool_result":
					marked.Role = RoleTool
					marked.ToolCallID = "call_1"
				}
				type block struct {
					Type  string        `json:"type"`
					Cache *CacheControl `json:"cache_control"`
				}
				var got struct {
					System   []block `json:"system"`
					Messages []struct {
						Content []block `json:"content"`
					} `json:"messages"`
				}
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
						t.Error(err)
					}
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = w.Write([]byte(anthropicStream("end_turn", true)))
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":1}}`))
				}))
				defer srv.Close()
				p := anthropicTestProvider(srv)
				msgs := []Message{{Role: RoleSystem, Content: "stable", CacheControl: &CacheControl{Type: "ephemeral"}}, {Role: RoleSystem, Content: "live"}, marked, {Role: RoleUser, Content: "tail"}}
				ctx := WithCacheSysPrompt(t.Context())
				var err error
				if stream {
					_, err = p.ChatStream(ctx, msgs, nil, func(StreamChunk) error { return nil })
				} else {
					_, err = p.Chat(ctx, msgs, nil)
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(got.System) != 2 || got.System[0].Cache == nil || got.System[1].Cache == nil {
					t.Fatal("lost system markers")
				}
				blocks := got.Messages[0].Content
				for i, b := range blocks {
					if (b.Cache != nil) != (i == len(blocks)-1) {
						t.Fatalf("wrong block marked: %+v", blocks)
					}
				}
				if blocks[len(blocks)-1].Type != kind {
					t.Fatalf("got %s want %s", blocks[len(blocks)-1].Type, kind)
				}
				if got.Messages[1].Content[0].Cache != nil {
					t.Fatal("marker leaked to unmarked tail")
				}
			})
		}
	}
}
