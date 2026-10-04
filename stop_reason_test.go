package ai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// sseServer answers every request with the given server-sent events.
func sseServer(t *testing.T, events string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(events))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func anthropicEvent(name, data string) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n", name, data)
}

// anthropicStream is a reply that says one line, finishes one tool call and
// starts another; stop is its stop reason, closeLast whether the second
// call's block gets its content_block_stop.
func anthropicStream(stop string, closeLast bool) string {
	var b strings.Builder
	b.WriteString(anthropicEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`))
	b.WriteString(anthropicEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
	b.WriteString(anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Writing both."}}`))
	b.WriteString(anthropicEvent("content_block_stop", `{"type":"content_block_stop","index":0}`))
	b.WriteString(anthropicEvent("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"write_file","input":{}}}`))
	b.WriteString(anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a.txt\",\"content\":\"one\"}"}}`))
	b.WriteString(anthropicEvent("content_block_stop", `{"type":"content_block_stop","index":1}`))
	b.WriteString(anthropicEvent("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_2","name":"write_file","input":{}}}`))
	b.WriteString(anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"b.txt\",\"content\":\"tw"}}`))
	if closeLast {
		b.WriteString(anthropicEvent("content_block_stop", `{"type":"content_block_stop","index":2}`))
	}
	b.WriteString(anthropicEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"`+stop+`","stop_sequence":null},"usage":{"output_tokens":64}}`))
	b.WriteString(anthropicEvent("message_stop", `{"type":"message_stop"}`))
	return b.String()
}

func anthropicTestProvider(srv *httptest.Server) *AnthropicProvider {
	return &AnthropicProvider{
		client: anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0)),
		model:  "claude-test",
	}
}

// A reply the output cap cut off says so, and the tool call it stopped in
// is set aside as Cut (never in ToolCalls): whether or not its block closed,
// its arguments are incomplete. The call before it finished and stays.
func TestAnthropicChatStream_CutOffReply(t *testing.T) {
	for _, closeLast := range []bool{false, true} {
		t.Run(fmt.Sprintf("block closed %v", closeLast), func(t *testing.T) {
			p := anthropicTestProvider(sseServer(t, anthropicStream("max_tokens", closeLast)))
			resp, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "write two files"}}, nil, func(StreamChunk) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if !resp.Truncated() || resp.StopReason != StopMaxTokens {
				t.Fatalf("stop reason %q, want %q", resp.StopReason, StopMaxTokens)
			}
			if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "toolu_1" {
				t.Fatalf("the finished call stays, the cut one goes: %+v", resp.ToolCalls)
			}
			if resp.Cut == nil || resp.Cut.ID != "toolu_2" || resp.Cut.Name != "write_file" || !strings.Contains(string(resp.Cut.Arguments), `"b.txt"`) {
				t.Fatalf("cut call = %+v", resp.Cut)
			}
			if resp.Content != "Writing both." {
				t.Fatalf("content %q", resp.Content)
			}
		})
	}
}

// A reply that ends on its own keeps every call and reports no cut.
func TestAnthropicChatStream_ToolUseStop(t *testing.T) {
	p := anthropicTestProvider(sseServer(t, anthropicStream("tool_use", true)))
	resp, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "write two files"}}, nil, func(StreamChunk) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if resp.Truncated() || resp.Cut != nil || resp.StopReason != "tool_use" || len(resp.ToolCalls) != 2 {
		t.Fatalf("stop %q, cut %+v, calls %d", resp.StopReason, resp.Cut, len(resp.ToolCalls))
	}
}

// The non-streamed call reads the same stop reason off the message.
func TestAnthropicChat_CutOffReply(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` +
			`{"type":"text","text":"Writing."},{"type":"tool_use","id":"toolu_1","name":"write_file","input":{"path":"a.txt"}}],` +
			`"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":64}}`))
	}))
	t.Cleanup(srv.Close)
	resp, err := anthropicTestProvider(srv).Chat(context.Background(), []Message{{Role: RoleUser, Content: "write a file"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Truncated() || len(resp.ToolCalls) != 0 || resp.Cut == nil || resp.Cut.Name != "write_file" {
		t.Fatalf("stop %q, calls %+v, cut %+v", resp.StopReason, resp.ToolCalls, resp.Cut)
	}
}

func openAIChunk(delta, finish string) string {
	f := "null"
	if finish != "" {
		f = `"` + finish + `"`
	}
	return `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + f + "}]}\n\n"
}

// An OpenAI-compatible stream that ends on finish_reason "length" is cut
// off: the call it stopped in is set aside.
func TestOpenAIChatStream_CutOffReply(t *testing.T) {
	for _, tt := range []struct {
		finish   string
		cut      bool
		stopWant string
	}{
		{"length", true, StopMaxTokens},
		{"tool_calls", false, "tool_calls"},
	} {
		t.Run(tt.finish, func(t *testing.T) {
			events := openAIChunk(`{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write_file","arguments":""}}]}`, "") +
				openAIChunk(`{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"a.txt\",\"content\":\"lo"}}]}`, "") +
				openAIChunk(`{}`, tt.finish) +
				"data: [DONE]\n\n"
			p := NewOpenAIProviderWithBaseURL("k", "gpt-test", sseServer(t, events).URL)
			resp, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "write a file"}}, nil, func(StreamChunk) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if resp.StopReason != tt.stopWant {
				t.Fatalf("stop reason %q, want %q", resp.StopReason, tt.stopWant)
			}
			if got := resp.Cut != nil; got != tt.cut {
				t.Fatalf("cut = %+v, want cut %v", resp.Cut, tt.cut)
			}
			if tt.cut && (len(resp.ToolCalls) != 0 || resp.Cut.Name != "write_file") {
				t.Fatalf("calls %+v, cut %+v", resp.ToolCalls, resp.Cut)
			}
			if !tt.cut && len(resp.ToolCalls) != 1 {
				t.Fatalf("a finished call stays: %+v", resp.ToolCalls)
			}
		})
	}
}
