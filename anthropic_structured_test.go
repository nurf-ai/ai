package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func TestAnthropicForcesToolChoice(t *testing.T) {
	for model, want := range map[string]bool{
		"claude-haiku-4-5":           true,
		"claude-haiku-4-5-20251001":  true,
		"claude-sonnet-4-6":          true,
		"claude-opus-4-8":            true,
		"claude-3-5-sonnet":          true,
		"claude-sonnet-5":            true,
		"claude-opus-5":              true,
		"claude-fable-5":             true,
		"claude-sonnet-5-5":          false,
		"claude-opus-5-5":            false,
		"claude-fable-5-1":           false,
		"claude-mythos-5-1":          false,
		"claude-some-future-model-9": false,
	} {
		if got := anthropicForcesToolChoice(model); got != want {
			t.Errorf("anthropicForcesToolChoice(%q) = %v, want %v", model, got, want)
		}
	}
}

// structuredServer answers every Messages call with content and records the
// last request body.
func structuredServer(t *testing.T, content string) (*httptest.Server, *map[string]any) {
	t.Helper()
	var req map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":` + content +
			`,"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &req
}

func structuredProvider(srv *httptest.Server, model string) *AnthropicProvider {
	return &AnthropicProvider{
		client: anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL), option.WithMaxRetries(0)),
		model:  model,
	}
}

var greetingSchema = json.RawMessage(`{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}`)

// Models that reject a forced tool_choice get auto (one call at most) plus a
// system nudge; the rest keep the forced structured_output call.
func TestAnthropicStructuredToolChoice(t *testing.T) {
	toolUse := `[{"type":"tool_use","id":"toolu_1","name":"structured_output","input":{"message":"hi"}}]`
	for _, tt := range []struct {
		model, choice string
		nudged        bool
	}{
		{"claude-sonnet-5-5", "auto", true},
		{"claude-haiku-4-5", "tool", false},
	} {
		t.Run(tt.model, func(t *testing.T) {
			srv, req := structuredServer(t, toolUse)
			p := structuredProvider(srv, tt.model)

			out, err := p.CreateStructuredOutputFromSchema(context.Background(), "greet me", "be brief", greetingSchema)
			if err != nil {
				t.Fatalf("from schema: %v", err)
			}
			if out["message"] != "hi" {
				t.Fatalf("message = %v, want hi", out["message"])
			}

			choice, _ := (*req)["tool_choice"].(map[string]any)
			if choice["type"] != tt.choice {
				t.Fatalf("tool_choice = %v, want type %q", choice, tt.choice)
			}
			if tt.choice == "auto" && choice["disable_parallel_tool_use"] != true {
				t.Errorf("auto tool_choice must cap the call count: %v", choice)
			}
			sys, _ := (*req)["system"].([]any)
			last, _ := sys[len(sys)-1].(map[string]any)
			if nudged := last["text"] == structuredOutputNudge; nudged != tt.nudged {
				t.Errorf("system = %v, nudged %v, want %v", sys, nudged, tt.nudged)
			}
			if first, _ := sys[0].(map[string]any); first["text"] != "be brief" {
				t.Errorf("caller system prompt must stay the first block: %v", sys)
			}

			var typed struct {
				Message string `json:"message"`
			}
			if err := p.CreateStructuredOutput(context.Background(), "greet me", "be brief", &typed); err != nil || typed.Message != "hi" {
				t.Fatalf("typed: %+v, err %v", typed, err)
			}
			if err := p.CreateStructuredOutputBreakpointed(context.Background(), "be brief", "mid", "greet me", &typed); err != nil {
				t.Fatalf("breakpointed: %v", err)
			}
			choice, _ = (*req)["tool_choice"].(map[string]any)
			if choice["type"] != tt.choice {
				t.Fatalf("breakpointed tool_choice = %v, want type %q", choice, tt.choice)
			}
		})
	}
}

// An unforced model may answer with the JSON in text instead of the call.
func TestAnthropicStructuredTextFallback(t *testing.T) {
	text, _ := json.Marshal("Here it is:\n```json\n{\"message\": \"hi\"}\n```")
	srv, _ := structuredServer(t, `[{"type":"text","text":`+string(text)+`}]`)
	out, err := structuredProvider(srv, "claude-sonnet-5-5").CreateStructuredOutputFromSchema(context.Background(), "greet me", "", greetingSchema)
	if err != nil {
		t.Fatalf("text fallback: %v", err)
	}
	if out["message"] != "hi" {
		t.Fatalf("message = %v, want hi", out["message"])
	}

	srv, _ = structuredServer(t, `[{"type":"text","text":"Which greeting would you like?"}]`)
	_, err = structuredProvider(srv, "claude-sonnet-5-5").CreateStructuredOutputFromSchema(context.Background(), "greet me", "", greetingSchema)
	if err == nil || !strings.Contains(err.Error(), "without calling structured_output") {
		t.Fatalf("prose answer: err = %v, want the no-call error", err)
	}
}
