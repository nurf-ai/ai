package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"
)

// fakeGemini answers generateContent with one scripted reply per call and
// keeps every request body.
func fakeGemini(t *testing.T, replies ...string) (*GeminiProvider, func() []map[string]any) {
	t.Helper()
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, ":generateContent") {
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		n := len(bodies)
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n >= len(replies) {
			http.Error(w, `{"error":{"code":500,"message":"no more replies"}}`, http.StatusInternalServerError)
			return
		}
		_, _ = io.WriteString(w, replies[n])
	}))
	t.Cleanup(srv.Close)
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey: "test", Backend: genai.BackendGeminiAPI, HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &GeminiProvider{client: client, model: "gemini-test"}, func() []map[string]any {
		mu.Lock()
		defer mu.Unlock()
		return bodies
	}
}

const geminiTwoCalls = `{"candidates":[{"content":{"role":"model","parts":[
 {"functionCall":{"name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"c2lnLTE="},
 {"functionCall":{"name":"get_time","args":{"city":"Paris"}}}
]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`

const geminiText = `{"candidates":[{"content":{"role":"model","parts":[{"text":"Sunny, 21 degrees, 3 pm."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":6}}`

// A tool round trip: the calls come back with their thought signatures,
// each result names the function it answers, and both results go back in
// one content (Gemini's rule for parallel calls). A call Gemini sent with no
// id gets one on this side that is never sent back.
func TestGemini_ToolRoundTripKeepsThoughtSignatures(t *testing.T) {
	p, bodies := fakeGemini(t, geminiTwoCalls, geminiText)
	tools := []Tool{{Name: "get_weather", Parameters: map[string]any{"type": "object"}}, {Name: "get_time", Parameters: map[string]any{"type": "object"}}}
	msgs := []Message{{Role: RoleSystem, Content: "be brief"}, {Role: RoleUser, Content: "weather and time in Paris?"}}

	res, err := p.Chat(context.Background(), msgs, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ToolCalls) != 2 || string(res.ToolCalls[0].ThoughtSignature) != "sig-1" || res.ToolCalls[1].ThoughtSignature != nil {
		t.Fatalf("calls: %+v", res.ToolCalls)
	}
	if !strings.HasPrefix(res.ToolCalls[0].ID, geminiCallPrefix) || res.ToolCalls[0].ID == res.ToolCalls[1].ID {
		t.Fatalf("id-less calls need ids of their own: %q %q", res.ToolCalls[0].ID, res.ToolCalls[1].ID)
	}

	msgs = append(msgs,
		Message{Role: RoleAssistant, ToolCalls: res.ToolCalls},
		Message{Role: RoleTool, ToolCallID: res.ToolCalls[0].ID, Content: `{"sky":"sunny","celsius":21}`},
		Message{Role: RoleTool, ToolCallID: res.ToolCalls[1].ID, Content: `3 pm`},
	)
	res, err = p.Chat(context.Background(), msgs, tools)
	if err != nil || res.Content != "Sunny, 21 degrees, 3 pm." {
		t.Fatalf("second step: %+v %v", res, err)
	}

	second := bodies()[1]
	contents, _ := second["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("want user, model calls, user results; got %d contents: %v", len(contents), contents)
	}
	model := contents[1].(map[string]any)
	calls := model["parts"].([]any)
	first := calls[0].(map[string]any)
	if first["thoughtSignature"] != "c2lnLTE=" || first["functionCall"].(map[string]any)["name"] != "get_weather" {
		t.Fatalf("the call went back without its signature: %v", first)
	}
	if id, ok := first["functionCall"].(map[string]any)["id"]; ok && id != "" {
		t.Fatalf("a made-up id was sent to Gemini: %v", id)
	}
	results := contents[2].(map[string]any)["parts"].([]any)
	if len(results) != 2 {
		t.Fatalf("both results go back in one content: %v", results)
	}
	for i, want := range []string{"get_weather", "get_time"} {
		fr := results[i].(map[string]any)["functionResponse"].(map[string]any)
		if fr["name"] != want {
			t.Errorf("result %d names %v, want %s", i, fr["name"], want)
		}
	}
	if sys, _ := second["systemInstruction"].(map[string]any); sys == nil {
		t.Error("the system instruction went missing")
	}
}

// A result whose call is not in the history keeps the old naming (its id),
// so a caller that only keeps results still sends something.
func TestGeminiContents_OrphanResult(t *testing.T) {
	contents, _ := geminiContents([]Message{{Role: RoleTool, ToolCallID: "get_weather", Content: "ok"}})
	if len(contents) != 1 || contents[0].Parts[0].FunctionResponse.Name != "get_weather" {
		t.Fatalf("orphan result: %+v", contents)
	}
}
