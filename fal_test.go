package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// newFalTestServer fakes the fal queue: POST /{endpoint} → ticket, GET
// status (IN_QUEUE once, then COMPLETED), GET result → output.
func newFalTestServer(t *testing.T, output any, onSubmit func(r *http.Request, body map[string]any)) (*httptest.Server, *int32) {
	t.Helper()
	var polls int32
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	base := srv.URL
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Key test-key" {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"detail":"bad key"}`))
			return
		}
		switch {
		case r.Method == http.MethodPost:
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode submit body: %v", err)
			}
			if onSubmit != nil {
				onSubmit(r, body)
			}
			ep := strings.Trim(r.URL.Path, "/")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"request_id":     "req-1",
				"status_url":     base + "/" + ep + "/requests/req-1/status",
				"response_url":   base + "/" + ep + "/requests/req-1",
				"cancel_url":     base + "/" + ep + "/requests/req-1/cancel",
				"queue_position": 3,
			})
		case strings.HasSuffix(r.URL.Path, "/status"):
			n := atomic.AddInt32(&polls, 1)
			st := "COMPLETED"
			if n == 1 {
				st = "IN_QUEUE"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"status": st, "queue_position": 0})
		case strings.Contains(r.URL.Path, "/requests/"):
			if output == nil {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"detail":"runner exploded"}`))
				return
			}
			_ = json.NewEncoder(w).Encode(output)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	return srv, &polls
}

func TestFalClient_Run(t *testing.T) {
	srv, polls := newFalTestServer(t, map[string]any{"ok": true}, nil)
	defer srv.Close()

	c := NewFalClient("test-key", WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
	raw, err := c.Run(context.Background(), "fal-ai/test/model", map[string]any{"prompt": "hi"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if string(raw) != "{\"ok\":true}\n" && strings.TrimSpace(string(raw)) != `{"ok":true}` {
		t.Fatalf("unexpected output %q", raw)
	}
	if atomic.LoadInt32(polls) != 2 {
		t.Fatalf("expected 2 status polls, got %d", *polls)
	}
}

func TestFalClient_ResultError(t *testing.T) {
	srv, _ := newFalTestServer(t, nil, nil)
	defer srv.Close()

	c := NewFalClient("test-key", WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
	_, err := c.Run(context.Background(), "fal-ai/test/model", map[string]any{})
	if err == nil {
		t.Fatal("expected error")
	}
	var fe *FalError
	if !errors.As(err, &fe) {
		t.Fatalf("expected *FalError, got %T: %v", err, err)
	}
	if fe.Status != 500 || fe.Message != "runner exploded" {
		t.Fatalf("unexpected FalError: %+v", fe)
	}
	if kind, _ := ClassifyError(err); kind != ErrProviderDown {
		t.Fatalf("expected ErrProviderDown, got %v", kind)
	}
}

func TestFalClient_AuthError(t *testing.T) {
	srv, _ := newFalTestServer(t, map[string]any{}, nil)
	defer srv.Close()

	c := NewFalClient("wrong", WithFalQueueBase(srv.URL))
	_, err := c.Submit(context.Background(), "fal-ai/test/model", map[string]any{})
	if kind, _ := ClassifyError(err); kind != ErrAuth {
		t.Fatalf("expected ErrAuth, got %v (%v)", kind, err)
	}
}

func TestFalErrorMessage(t *testing.T) {
	cases := map[string]string{
		`{"detail":"nope"}`: "nope",
		`{"detail":[{"loc":["body","image_url"],"msg":"field required","type":"value_error"}]}`: "body.image_url: field required",
		`{"error":"boom"}`: "boom",
		`plain text`:       "plain text",
		``:                 "no error detail",
	}
	for in, want := range cases {
		if got := falErrorMessage([]byte(in)); got != want {
			t.Errorf("falErrorMessage(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFalClient_WaitCancelsOnCtx(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	var cancelled int32
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost:
			_ = json.NewEncoder(w).Encode(map[string]any{
				"request_id": "r", "status_url": srv.URL + "/m/requests/r/status",
				"response_url": srv.URL + "/m/requests/r", "cancel_url": srv.URL + "/m/requests/r/cancel",
			})
		case strings.HasSuffix(r.URL.Path, "/cancel"):
			atomic.StoreInt32(&cancelled, 1)
		default:
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "IN_PROGRESS"})
		}
	})
	c := NewFalClient("k", WithFalQueueBase(srv.URL), WithFalPollInterval(5*time.Millisecond))
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	_, err := c.Run(ctx, "m", map[string]any{})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline error, got %v", err)
	}
	if atomic.LoadInt32(&cancelled) != 1 {
		t.Fatal("expected cancel to be called")
	}
}

func TestFalVideoProvider_Generate(t *testing.T) {
	var got map[string]any
	var gotPath string
	output := map[string]any{
		"video": map[string]any{
			"url": "https://v3.fal.media/files/x/out.mp4", "content_type": "video/mp4",
			"width": 1920, "height": 1080, "fps": 25, "duration": 6, "num_frames": 150, "file_size": 1234,
		},
		"seed": 42,
	}
	srv, _ := newFalTestServer(t, output, func(r *http.Request, body map[string]any) {
		got = body
		gotPath = r.URL.Path
	})
	defer srv.Close()

	client := NewFalClient("test-key", WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
	p := NewFalVideoProviderWithClient(client, "")
	var events []UsageEvent
	p.SetMeter(func(ev UsageEvent) { events = append(events, ev) })

	seed := int64(7)
	res, err := p.Generate(WithMeterOperation(context.Background(), "tv"), VideoRequest{
		Prompt:      "a cat surfing",
		Image:       []byte("jpegbytes"),
		Duration:    6,
		Resolution:  "1080p",
		AspectRatio: "16:9",
		Seed:        &seed,
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotPath != "/"+DefaultFalVideoModel {
		t.Fatalf("submitted to %q, want default model", gotPath)
	}
	if got["prompt"] != "a cat surfing" {
		t.Errorf("prompt not forwarded: %v", got["prompt"])
	}
	if u, _ := got["image_url"].(string); !strings.HasPrefix(u, "data:image/jpeg;base64,") {
		t.Errorf("image_url should be a jpeg data URI, got %q", u)
	}
	if d, _ := got["duration"].(float64); d != 6 {
		t.Errorf("duration = %v, want 6", got["duration"])
	}
	if got["generate_audio"] != false {
		t.Errorf("generate_audio = %v, want false", got["generate_audio"])
	}
	if s, _ := got["seed"].(float64); s != 7 {
		t.Errorf("seed = %v, want 7", got["seed"])
	}
	if res.URL != "https://v3.fal.media/files/x/out.mp4" || res.Width != 1920 || res.Duration != 6 || res.Seed != 42 {
		t.Fatalf("unexpected result: %+v", res)
	}
	if res.CostUSD != 0.24 {
		t.Errorf("cost = %v, want 0.24 (6s @ $0.04)", res.CostUSD)
	}
	if len(events) != 1 || events[0].Provider != "fal" || events[0].Operation != "tv" || events[0].Metadata["type"] != "video_gen" {
		t.Fatalf("unexpected meter events: %+v", events)
	}
}

func TestFalVideoProvider_ModelOverrideAndImageURL(t *testing.T) {
	var got map[string]any
	var gotPath string
	srv, _ := newFalTestServer(t, map[string]any{"video": map[string]any{"url": "u"}}, func(r *http.Request, body map[string]any) {
		got = body
		gotPath = r.URL.Path
	})
	defer srv.Close()
	client := NewFalClient("test-key", WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
	p := NewFalVideoProviderWithClient(client, "fal-ai/custom/i2v")

	res, err := p.Generate(context.Background(), VideoRequest{Prompt: "p", ImageURL: "https://x/frame.png", Model: "fal-ai/other/t2v", Duration: 8})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotPath != "/fal-ai/other/t2v" {
		t.Errorf("model override not used: %q", gotPath)
	}
	if got["image_url"] != "https://x/frame.png" {
		t.Errorf("image_url = %v", got["image_url"])
	}
	if res.Duration != 8 {
		t.Errorf("duration fallback = %v, want 8 (request)", res.Duration)
	}
	if res.ContentType != "video/mp4" {
		t.Errorf("content type default = %q", res.ContentType)
	}
}

func TestEstimateVideoCost(t *testing.T) {
	if got := EstimateVideoCost("fal-ai/ltx-2.3/image-to-video/fast", 6, "1080p"); got != 0.24 {
		t.Errorf("1080p: got %v want 0.24", got)
	}
	if got := EstimateVideoCost("fal-ai/ltx-2.3/image-to-video/fast", 6, "2160p"); got != 0.96 {
		t.Errorf("2160p: got %v want 0.96", got)
	}
	if got := EstimateVideoCost("fal-ai/ltx-2.3/image-to-video/fast", 6, ""); got != 0.24 {
		t.Errorf("default res: got %v want 0.24", got)
	}
	if got := EstimateVideoCost("nope", 6, "1080p"); got != 0 {
		t.Errorf("unknown: got %v want 0", got)
	}
	if !IsVideoModel("fal-ai/ltx-2.3/text-to-video/fast") || IsVideoModel("gpt-4o") {
		t.Error("IsVideoModel misclassified")
	}
}

func TestNewVideoProvider(t *testing.T) {
	p, err := NewVideoProvider("fal", "k", "")
	if err != nil || p == nil || p.Name() != "fal" || p.Model() != DefaultFalVideoModel {
		t.Fatalf("fal provider: %v %+v", err, p)
	}
	if _, err := NewVideoProvider("nope", "k", ""); err == nil {
		t.Fatal("expected unsupported provider error")
	}
}

func TestDataURI(t *testing.T) {
	if got := DataURI("image/png", []byte{1, 2, 3}); got != "data:image/png;base64,AQID" {
		t.Errorf("DataURI = %q", got)
	}
}
