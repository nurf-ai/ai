package ai

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// A zero-argument tool call streams no input_json_delta: its arguments come
// back "". Sent back as-is in the next request's tool_use block, that fails
// to marshal and the whole turn dies.
func TestToolInput_EmptyArgsMarshal(t *testing.T) {
	for _, args := range []string{"", "  \n"} {
		msg := anthropic.MessageParam{Role: "assistant", Content: []anthropic.ContentBlockParamUnion{
			{OfToolUse: &anthropic.ToolUseBlockParam{ID: "toolu_1", Name: "get_time", Input: toolInput(json.RawMessage(args))}},
		}}
		b, err := json.Marshal(msg)
		if err != nil {
			t.Fatalf("args %q: marshal: %v", args, err)
		}
		var back struct {
			Content []struct {
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		}
		if err := json.Unmarshal(b, &back); err != nil || len(back.Content) != 1 || string(back.Content[0].Input) != "{}" {
			t.Fatalf("args %q: input = %s (err %v), want {}", args, b, err)
		}
	}
	if got := toolInput(json.RawMessage(`{"text":"hi"}`)); string(got) != `{"text":"hi"}` {
		t.Fatalf("non-empty args rewritten: %s", got)
	}
}
