package ai

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
)

func jsonServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// An embeddings call is metered on its own: the API's token count, the
// embedding model's price, operation "embeddings" whatever the caller
// stamped, and the caller on the ctx.
func TestEmbedText_Metered(t *testing.T) {
	srv := jsonServer(t, `{"object":"list","data":[{"object":"embedding","embedding":[0.1,0.2],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":800,"total_tokens":800}}`)
	p := NewOpenAIProviderWithBaseURL("k", "gpt-test", srv.URL)
	tap := &meterTap{}
	SetEmbedderMeter(p, tap.hook)
	caller := uuid.New()
	ctx := WithMeterOperation(WithMeterCallerID(context.Background(), caller), "app.llm")
	if _, err := p.EmbedText(ctx, []string{"a file"}); err != nil {
		t.Fatal(err)
	}
	ev := tap.one(t)
	if ev.CallerID != caller || ev.Operation != "embeddings" || ev.Model != "text-embedding-3-small" || ev.InputTokens != 800 {
		t.Fatalf("event = %+v", ev)
	}
	if want := 800 * 0.02 / 1_000_000; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", ev.EstimatedCostUSD, want)
	}
}

func sttProvider(url, model string) *OpenAISTTProvider {
	cfg := openai.DefaultConfig("k")
	cfg.BaseURL = url
	return &OpenAISTTProvider{raw: openai.NewClientWithConfig(cfg), model: model}
}

// A transcription is priced by minute of audio: the duration whisper-1
// reports, or one estimated from the upload's size.
func TestTranscribe_Metered(t *testing.T) {
	caller := uuid.New()
	ctx := WithMeterCallerID(context.Background(), caller)

	whisper := sttProvider(jsonServer(t, `{"task":"transcribe","language":"english","duration":30,"text":"hello"}`).URL, openai.Whisper1)
	tap := &meterTap{}
	SetSTTMeter(whisper, tap.hook)
	if text, err := whisper.Transcribe(ctx, strings.NewReader("audio"), "a.webm"); err != nil || text != "hello" {
		t.Fatalf("text %q err %v", text, err)
	}
	ev := tap.one(t)
	if ev.CallerID != caller || ev.Operation != "stt" || ev.Metadata["seconds"] != 30.0 || ev.Metadata["estimated"] != nil {
		t.Fatalf("event = %+v", ev)
	}
	if want := 0.006 * 30 / 60; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", ev.EstimatedCostUSD, want)
	}

	// no duration in the answer: 8 KB of browser Opus is about 2 s
	mini := sttProvider(jsonServer(t, `{"text":"hello"}`).URL, "gpt-4o-mini-transcribe")
	tap = &meterTap{}
	SetSTTMeter(mini, tap.hook)
	if _, err := mini.Transcribe(ctx, strings.NewReader(strings.Repeat("x", 8000)), "a.webm"); err != nil {
		t.Fatal(err)
	}
	ev = tap.one(t)
	if ev.Metadata["estimated"] != true || ev.Metadata["seconds"] != 2.0 {
		t.Fatalf("event = %+v", ev)
	}
}
