package ai

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTextToVideoEndpoint(t *testing.T) {
	cases := map[string]string{
		"fal-ai/ltx-2.3/image-to-video/fast": "fal-ai/ltx-2.3/text-to-video/fast",
		"fal-ai/ltx-2.3/text-to-video/fast":  "fal-ai/ltx-2.3/text-to-video/fast",
		"fal-ai/minimax/hailuo-02/standard":  "fal-ai/minimax/hailuo-02/standard",
	}
	for in, want := range cases {
		if got := textToVideoEndpoint(in); got != want {
			t.Errorf("textToVideoEndpoint(%q) = %q, want %q", in, got, want)
		}
	}
}

// Without a conditioning frame the image-to-video endpoint would 422
// (image_url required) — the provider must submit to text-to-video instead,
// and keep image-to-video when a frame is given.
func TestFalVideoProvider_TextToVideoWhenNoImage(t *testing.T) {
	var mu sync.Mutex
	var paths []string
	var bodies []map[string]any
	output := map[string]any{"video": map[string]any{"url": "https://cdn.test/clip.mp4", "content_type": "video/mp4", "duration": 6.0}, "seed": 7}
	srv, _ := newFalTestServer(t, output, func(r *http.Request, body map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, body)
	})
	defer srv.Close()
	client := NewFalClient("test-key", WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
	p := NewFalVideoProviderWithClient(client, "fal-ai/ltx-2.3/image-to-video/fast")

	res, err := p.Generate(context.Background(), VideoRequest{Prompt: "a dog plays violin", Duration: 6})
	if err != nil {
		t.Fatalf("text-to-video generate: %v", err)
	}
	if !strings.Contains(res.Model, "text-to-video") {
		t.Fatalf("res.Model = %q, want text-to-video endpoint", res.Model)
	}
	if _, err := p.Generate(context.Background(), VideoRequest{Prompt: "same, from a frame", ImageURL: "https://cdn.test/frame.jpg"}); err != nil {
		t.Fatalf("image-to-video generate: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 2 {
		t.Fatalf("submits = %d, want 2 (%v)", len(paths), paths)
	}
	if !strings.Contains(paths[0], "/text-to-video/") || bodies[0]["image_url"] != nil {
		t.Errorf("no-image submit: path %q body %v — want text-to-video without image_url", paths[0], bodies[0])
	}
	if !strings.Contains(paths[1], "/image-to-video/") || bodies[1]["image_url"] != "https://cdn.test/frame.jpg" {
		t.Errorf("image submit: path %q body %v — want image-to-video with image_url", paths[1], bodies[1])
	}
}

// The LTX-Video 0.9 family counts frames and has no audio knob; the 13B
// distilled endpoint keeps resolution/aspect, plain ltx-video takes prompt only.
func TestFalVideoProvider_LTXVideoFamilyInput(t *testing.T) {
	p := NewFalVideoProviderWithClient(NewFalClient("k"), "")
	in := p.buildInput("fal-ai/ltxv-13b-098-distilled", VideoRequest{Prompt: "x", Duration: 5, Resolution: "720p", AspectRatio: "16:9", Audio: true})
	if in["num_frames"] != 121 || in["frame_rate"] != 24 || in["resolution"] != "720p" || in["aspect_ratio"] != "16:9" {
		t.Errorf("ltxv-13b input = %v", in)
	}
	for _, k := range []string{"duration", "generate_audio", "fps"} {
		if _, ok := in[k]; ok {
			t.Errorf("ltxv-13b input leaks %q: %v", k, in)
		}
	}
	in = p.buildInput("fal-ai/ltxv-13b-098-distilled", VideoRequest{Prompt: "x", Resolution: "1080p"})
	if _, ok := in["resolution"]; ok {
		t.Errorf("unsupported 1080p should fall back to the endpoint default: %v", in)
	}
	in = p.buildInput("fal-ai/ltx-video", VideoRequest{Prompt: "x", Duration: 6, Resolution: "1080p", AspectRatio: "16:9"})
	if len(in) != 1 || in["prompt"] != "x" {
		t.Errorf("ltx-video input should be prompt only: %v", in)
	}
}
