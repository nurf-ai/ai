package ai

import (
	"context"
	"errors"
	"io"
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

// sttCall is what the fake transcription endpoint received.
type sttCall struct {
	path, auth, model, format, filename string
	file                                []byte
}

// sttFake is a provider for model whose endpoint answers status and body,
// and the channel the request it received arrives on.
func sttFake(t *testing.T, model string, status int, body string) (*OpenAISTTProvider, <-chan sttCall) {
	t.Helper()
	calls := make(chan sttCall, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := sttCall{path: r.URL.Path, auth: r.Header.Get("Authorization")}
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			t.Errorf("multipart form: %v", err)
		}
		c.model, c.format = r.FormValue("model"), r.FormValue("response_format")
		if f, h, err := r.FormFile("file"); err == nil {
			c.filename = h.Filename
			c.file, _ = io.ReadAll(f)
			_ = f.Close()
		}
		select {
		case calls <- c:
		default:
			t.Error("more than one request")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	p := newOpenAISTTProvider("k", model)
	p.baseURL = srv.URL
	return p, calls
}

// gpt-4o-transcribe bills by token: the event carries the tokens the API
// reported and their price, and the seconds come from the audio tokens, not
// from the upload's size (a clip recorded above 32 kbps counted, and cost,
// more than its length).
func TestTranscribe_TokenUsage(t *testing.T) {
	p, calls := sttFake(t, "gpt-4o-transcribe", http.StatusOK,
		`{"text":"hello","usage":{"type":"tokens","input_tokens":620,"input_token_details":{"text_tokens":20,"audio_tokens":600},"output_tokens":150,"total_tokens":770}}`)
	tap := &meterTap{}
	SetSTTMeter(p, tap.hook)
	caller := uuid.New()
	clip := strings.Repeat("x", 480_000) // 120 s by its size, 60 s by its 600 audio tokens
	text, err := p.Transcribe(WithMeterCallerID(context.Background(), caller), strings.NewReader(clip), "hello.webm")
	if err != nil || text != "hello" {
		t.Fatalf("text %q err %v", text, err)
	}
	if c := <-calls; c.path != "/audio/transcriptions" || c.auth != "Bearer k" || c.model != "gpt-4o-transcribe" ||
		c.format != "" || c.filename != "hello.webm" || string(c.file) != clip {
		t.Fatalf("request: path %q auth %q model %q format %q filename %q, %d bytes of file",
			c.path, c.auth, c.model, c.format, c.filename, len(c.file))
	}
	ev := tap.one(t)
	if ev.CallerID != caller || ev.Operation != "stt" || ev.Model != "gpt-4o-transcribe" ||
		ev.InputTokens != 620 || ev.OutputTokens != 150 || ev.TotalTokens != 770 {
		t.Fatalf("event = %+v", ev)
	}
	md := ev.Metadata
	if md["seconds"] != 60.0 || md["seconds_from"] != "audio_tokens" || md["estimated"] != nil ||
		md["audio_input_tok"] != 600 || md["text_input_tok"] != 20 {
		t.Fatalf("metadata = %v", md)
	}
	// audio in at $6/M, text in at $2.50/M, out at $10/M
	if want := (600*6.0 + 20*2.5 + 150*10.0) / 1_000_000; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", ev.EstimatedCostUSD, want)
	}
}

// Tokens from a model with no token prices still give the seconds (every
// input token is audio when the API sends no breakdown), priced per minute.
func TestTranscribe_TokenUsagePricedPerMinute(t *testing.T) {
	p, _ := sttFake(t, "gpt-transcribe", http.StatusOK,
		`{"text":"hello","usage":{"type":"tokens","input_tokens":300,"output_tokens":40,"total_tokens":340}}`)
	tap := &meterTap{}
	SetSTTMeter(p, tap.hook)
	if _, err := p.Transcribe(context.Background(), strings.NewReader("audio"), "hello.webm"); err != nil {
		t.Fatal(err)
	}
	ev := tap.one(t)
	if ev.InputTokens != 300 || ev.Metadata["seconds"] != 30.0 || ev.Metadata["seconds_from"] != "audio_tokens" || ev.Metadata["audio_input_tok"] != 300 {
		t.Fatalf("event = %+v", ev)
	}
	if want := 0.0045 * 30 / 60; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
		t.Fatalf("cost = %v, want %v", ev.EstimatedCostUSD, want)
	}
}

// whisper-1 bills by minute of audio: it is asked for verbose_json, and the
// seconds the API bills set the price; an answer with only the verbose
// duration is priced on that.
func TestTranscribe_DurationUsage(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"usage", `{"task":"transcribe","language":"english","duration":29.4,"text":"hello","usage":{"type":"duration","seconds":30}}`},
		{"duration", `{"task":"transcribe","language":"english","duration":30,"text":"hello"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, calls := sttFake(t, openai.Whisper1, http.StatusOK, tc.body)
			tap := &meterTap{}
			SetSTTMeter(p, tap.hook)
			if text, err := p.Transcribe(context.Background(), strings.NewReader("audio"), "hello.webm"); err != nil || text != "hello" {
				t.Fatalf("text %q err %v", text, err)
			}
			if c := <-calls; c.model != "whisper-1" || c.format != "verbose_json" || c.filename != "hello.webm" || string(c.file) != "audio" {
				t.Fatalf("request: model %q format %q filename %q file %q", c.model, c.format, c.filename, c.file)
			}
			ev := tap.one(t)
			if ev.InputTokens != 0 || ev.Metadata["seconds"] != 30.0 || ev.Metadata["seconds_from"] != "duration" || ev.Metadata["estimated"] != nil {
				t.Fatalf("event = %+v", ev)
			}
			if want := 0.006 * 30 / 60; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
				t.Fatalf("cost = %v, want %v", ev.EstimatedCostUSD, want)
			}
		})
	}
}

// With no usage to read only the upload's size is left: 8 KB of browser Opus
// is about 2 s, marked estimated. A usage this code cannot read is metered
// the same way, and the transcript still comes back.
func TestTranscribe_NoUsage(t *testing.T) {
	for _, body := range []string{`{"text":"hello"}`, `{"text":"hello","usage":"unexpected"}`} {
		p, _ := sttFake(t, "gpt-4o-mini-transcribe", http.StatusOK, body)
		tap := &meterTap{}
		SetSTTMeter(p, tap.hook)
		if text, err := p.Transcribe(context.Background(), strings.NewReader(strings.Repeat("x", 8000)), "hello.webm"); err != nil || text != "hello" {
			t.Fatalf("%s: text %q err %v", body, text, err)
		}
		ev := tap.one(t)
		if ev.InputTokens != 0 || ev.Metadata["seconds"] != 2.0 || ev.Metadata["seconds_from"] != "size" || ev.Metadata["estimated"] != true {
			t.Fatalf("%s: event = %+v", body, ev)
		}
		if want := 0.003 * 2 / 60; math.Abs(ev.EstimatedCostUSD-want) > 1e-12 {
			t.Fatalf("%s: cost = %v, want %v", body, ev.EstimatedCostUSD, want)
		}
	}
}

// A refused call comes back as go-openai's error, so ClassifyError still
// reads it, and nothing is metered.
func TestTranscribe_APIError(t *testing.T) {
	p, _ := sttFake(t, "gpt-4o-transcribe", http.StatusTooManyRequests,
		`{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}`)
	tap := &meterTap{}
	SetSTTMeter(p, tap.hook)
	_, err := p.Transcribe(context.Background(), strings.NewReader("audio"), "hello.webm")
	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) || apiErr.HTTPStatusCode != http.StatusTooManyRequests || apiErr.Message != "Rate limit reached" {
		t.Fatalf("err = %v", err)
	}
	if kind, _ := ClassifyError(err); kind != ErrRateLimit {
		t.Fatalf("kind = %v, want rate limit", kind)
	}
	if len(tap.events) != 0 {
		t.Fatalf("metered a refused call: %+v", tap.events)
	}

	// a body that is not the API's error object
	p, _ = sttFake(t, "gpt-4o-transcribe", http.StatusBadGateway, `<html>bad gateway</html>`)
	_, err = p.Transcribe(context.Background(), strings.NewReader("audio"), "hello.webm")
	var reqErr *openai.RequestError
	if !errors.As(err, &reqErr) || reqErr.HTTPStatusCode != http.StatusBadGateway || string(reqErr.Body) != `<html>bad gateway</html>` {
		t.Fatalf("err = %v", err)
	}

	if _, err := p.Transcribe(context.Background(), nil, "hello.webm"); err == nil {
		t.Fatal("no audio, no error")
	}
}
