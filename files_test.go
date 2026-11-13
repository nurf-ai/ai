package ai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

var (
	testPDF = []byte("%PDF-1.4 fixture")
	testWAV = []byte("RIFF fixture")
)

func fileMsg(parts ...Part) []Message {
	return []Message{{Role: RoleUser, Parts: parts}}
}

func TestFilePartJSONRoundTrip(t *testing.T) {
	in := Message{Role: RoleUser, Parts: []Part{
		TextPart{Text: "read this"},
		FilePart{MediaType: "application/pdf", Data: testPDF, Name: "report.pdf"},
		FilePart{MediaType: "audio/wav", Data: testWAV},
	}}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Parts []map[string]any `json:"parts"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Parts[1]["type"] != "file" || wire.Parts[1]["name"] != "report.pdf" ||
		wire.Parts[1]["data"] != base64.StdEncoding.EncodeToString(testPDF) {
		t.Fatalf("wire shape: %v", wire.Parts[1])
	}
	var out Message
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("round trip:\n in %#v\nout %#v", in, out)
	}
	if err := json.Unmarshal([]byte(`{"role":"user","parts":[{"type":"file","data":"!!"}]}`), &out); err == nil {
		t.Fatal("bad base64 must fail, not decode to empty data")
	}
}

func TestAcceptsFileMatrix(t *testing.T) {
	gem := &GeminiProvider{}
	provs := map[string]FileAcceptor{
		"anthropic":   &AnthropicProvider{},
		"openai":      &OpenAIProvider{},
		"openrouter":  &OpenRouterProvider{},
		"huggingface": &HuggingFaceProvider{},
		"gemini":      gem,
		"ollama":      &OllamaProvider{},
	}
	types := []string{"application/pdf", "audio/wav", "audio/mpeg", "audio/ogg", "video/mp4", "image/png", "text/plain"}
	want := map[string][]bool{
		//             pdf    wav    mpeg   ogg    mp4    png    txt
		"anthropic":   {true, false, false, false, false, false, false},
		"openai":      {true, true, true, false, false, false, false},
		"openrouter":  {true, true, true, false, false, false, false},
		"huggingface": {true, true, true, false, false, false, false},
		"gemini":      {true, true, true, true, true, false, false},
		"ollama":      {false, false, false, false, false, false, false},
	}
	for name, p := range provs {
		for i, mt := range types {
			if got := p.AcceptsFile(mt); got != want[name][i] {
				t.Errorf("%s AcceptsFile(%q) = %v, want %v", name, mt, got, want[name][i])
			}
		}
	}
	if !provs["openai"].AcceptsFile("Application/PDF; charset=x") {
		t.Error("media type compare ignores case and parameters")
	}
	if !AcceptsFile(&OpenAIProvider{}, "application/pdf") {
		t.Error("package AcceptsFile uses the provider")
	}
	if AcceptsFile(&stubProvider{}, "application/pdf") {
		t.Error("a provider without FileAcceptor accepts nothing")
	}
}

func TestFileUnsupportedSentinel(t *testing.T) {
	ctx := context.Background()
	cases := map[string]LLMProvider{
		"anthropic":  &AnthropicProvider{},
		"openai":     &OpenAIProvider{},
		"openrouter": &OpenRouterProvider{},
		"gemini":     &GeminiProvider{},
		"ollama":     &OllamaProvider{},
	}
	// a type no provider takes, and (for ollama) a pdf
	for name, p := range cases {
		mt := "text/csv"
		if name == "ollama" {
			mt = "application/pdf"
		}
		msgs := fileMsg(FilePart{MediaType: mt, Data: testPDF})
		if _, err := p.Chat(ctx, msgs, nil); !errors.Is(err, ErrFileUnsupported) {
			t.Errorf("%s Chat: %v", name, err)
		}
		if sp, ok := p.(StreamingProvider); ok {
			if _, err := sp.ChatStream(ctx, msgs, nil, func(StreamChunk) error { return nil }); !errors.Is(err, ErrFileUnsupported) {
				t.Errorf("%s ChatStream: %v", name, err)
			}
		}
		if mp, ok := p.(MultimodalStructuredProvider); ok {
			if _, err := mp.CreateStructuredOutputFromParts(ctx, msgs[0].Parts, "", json.RawMessage(`{}`)); !errors.Is(err, ErrFileUnsupported) {
				t.Errorf("%s structured: %v", name, err)
			}
		}
	}
	// a provider with no multimodal path at all
	if _, err := StructuredOutputFromParts(ctx, &textOnlyStub{}, []Part{FilePart{MediaType: "application/pdf", Data: testPDF}}, "", nil); !errors.Is(err, ErrFileUnsupported) {
		t.Errorf("fallback: %v", err)
	}
}

func TestAnthropicDocumentBlock(t *testing.T) {
	blocks := anthropicPartBlocks([]Part{
		FilePart{MediaType: "application/pdf", Data: testPDF, Name: "report.pdf"},
		TextPart{Text: "summarize"},
	})
	b, _ := json.Marshal(blocks)
	var got []map[string]any
	_ = json.Unmarshal(b, &got)
	if got[0]["type"] != "document" || got[0]["title"] != "report.pdf" {
		t.Fatalf("block: %s", b)
	}
	src := got[0]["source"].(map[string]any)
	if src["type"] != "base64" || src["media_type"] != "application/pdf" || src["data"] != base64.StdEncoding.EncodeToString(testPDF) {
		t.Fatalf("source: %v", src)
	}
	if got[1]["type"] != "text" {
		t.Fatalf("text kept: %s", b)
	}
}

func TestAnthropicChatSendsDocument(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()
	p := NewAnthropicProvider("k", "claude-test")
	p.client = anthropic.NewClient(option.WithAPIKey("k"), option.WithBaseURL(srv.URL))
	if _, err := p.Chat(context.Background(), fileMsg(FilePart{MediaType: "application/pdf", Data: testPDF}), nil); err != nil {
		t.Fatal(err)
	}
	content := body["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["type"] != "document" {
		t.Fatalf("content: %v", content)
	}
}

// openaiBody runs msgs through p's Chat against a fake server and returns the
// content parts of the first user message as they arrived on the wire.
func openaiBody(t *testing.T, newP func(url string) LLMProvider, stream bool, parts ...Part) []map[string]any {
	t.Helper()
	var got []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if int64(len(raw)) != r.ContentLength {
			t.Errorf("content-length %d, body %d", r.ContentLength, len(raw))
		}
		var req struct {
			Messages []struct {
				Content []map[string]any `json:"content"`
			} `json:"messages"`
			Stream bool `json:"stream"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Errorf("body: %v", err)
		}
		got = req.Messages[0].Content
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()
	p := newP(srv.URL)
	var err error
	if stream {
		_, err = p.(StreamingProvider).ChatStream(context.Background(), fileMsg(parts...), nil, func(StreamChunk) error { return nil })
	} else {
		_, err = p.Chat(context.Background(), fileMsg(parts...), nil)
	}
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestOpenAIFileEncoding(t *testing.T) {
	providers := map[string]func(string) LLMProvider{
		"openai":      func(u string) LLMProvider { return NewOpenAIProviderWithBaseURL("k", "gpt-test", u) },
		"openrouter":  func(u string) LLMProvider { return newOpenRouterProvider("k", "m", u) },
		"huggingface": func(u string) LLMProvider { return NewHuggingFaceProvider("k", "m", u) },
	}
	for name, newP := range providers {
		for _, stream := range []bool{false, true} {
			got := openaiBody(t, newP, stream,
				TextPart{Text: "look"},
				FilePart{MediaType: "application/pdf", Data: testPDF, Name: "report.pdf"},
				FilePart{MediaType: "application/pdf", Data: testPDF},
				FilePart{MediaType: "audio/wav", Data: testWAV},
				FilePart{MediaType: "audio/mpeg", Data: testWAV},
				ImagePart{MediaType: "image/png", Data: "AAAA"},
			)
			b64pdf := base64.StdEncoding.EncodeToString(testPDF)
			b64aud := base64.StdEncoding.EncodeToString(testWAV)
			want := []map[string]any{
				{"type": "text", "text": "look"},
				{"type": "file", "file": map[string]any{"filename": "report.pdf", "file_data": "data:application/pdf;base64," + b64pdf}},
				{"type": "file", "file": map[string]any{"filename": "file.pdf", "file_data": "data:application/pdf;base64," + b64pdf}},
				{"type": "input_audio", "input_audio": map[string]any{"data": b64aud, "format": "wav"}},
				{"type": "input_audio", "input_audio": map[string]any{"data": b64aud, "format": "mp3"}},
				{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,AAAA"}},
			}
			if !reflect.DeepEqual(got, want) {
				gb, _ := json.MarshalIndent(got, "", " ")
				t.Errorf("%s stream=%v wire:\n%s", name, stream, gb)
			}
		}
	}
}

func TestOpenAIUserMessageFromPartsFile(t *testing.T) {
	m := openaiUserMessageFromParts([]Part{TextPart{Text: "x"}, FilePart{MediaType: "application/pdf", Data: testPDF}})
	if len(m.MultiContent) != 2 || m.Content != "" {
		t.Fatalf("a file forces the multi-part form: %+v", m)
	}
}

func TestFilePartTransportLeavesOtherBodies(t *testing.T) {
	in := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	var seen []byte
	rt := filePartTransport{base: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		seen, _ = io.ReadAll(r.Body)
		return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader(nil))}, nil
	})}
	req, _ := http.NewRequest("POST", "http://x/y", bytes.NewReader(in))
	if _, err := rt.RoundTrip(req); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(seen, in) {
		t.Fatalf("body changed: %s", seen)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestGeminiFileEncoding(t *testing.T) {
	contents, _ := geminiContents(fileMsg(
		FilePart{MediaType: "application/pdf", Data: testPDF},
		FilePart{MediaType: "audio/mpeg; codecs=x", Data: testWAV},
		FilePart{MediaType: "video/mp4", Data: testWAV},
		TextPart{Text: "go"},
	))
	parts := contents[0].Parts
	wantMT := []string{"application/pdf", "audio/mpeg", "video/mp4"}
	for i, mt := range wantMT {
		if parts[i].InlineData == nil || parts[i].InlineData.MIMEType != mt {
			t.Fatalf("part %d: %+v", i, parts[i])
		}
	}
	if !bytes.Equal(parts[0].InlineData.Data, testPDF) {
		t.Error("blob carries the raw bytes")
	}
	if parts[3].Text != "go" {
		t.Error("text kept")
	}
}

func TestGeminiInlineLimit(t *testing.T) {
	p := &GeminiProvider{}
	big := FilePart{MediaType: "video/mp4", Data: make([]byte, geminiInlineLimit/2+1)}
	_, err := p.Chat(context.Background(), fileMsg(big, big), nil)
	if !errors.Is(err, ErrFileTooLarge) {
		t.Fatalf("want ErrFileTooLarge, got %v", err)
	}
	if err := checkFileParts(fileMsg(FilePart{MediaType: "video/mp4", Data: make([]byte, geminiInlineLimit)}), geminiAcceptsFile, geminiInlineLimit); err != nil {
		t.Fatalf("exactly the limit is allowed: %v", err)
	}
}
