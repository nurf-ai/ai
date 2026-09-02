package ai

import (
	"context"
	"fmt"
	"testing"
	"time"
)

type stubVideoProvider struct {
	name    string
	model   string
	cost    float64
	elapsed time.Duration
}

func (s *stubVideoProvider) Name() string  { return s.name }
func (s *stubVideoProvider) Model() string { return s.model }
func (s *stubVideoProvider) Generate(_ context.Context, _ VideoRequest) (*VideoResult, error) {
	return &VideoResult{
		URL:     "https://example.com/" + s.name + ".mp4",
		Model:   s.model,
		CostUSD: s.cost,
		Elapsed: s.elapsed,
	}, nil
}

type failingVideoProvider struct {
	name  string
	model string
}

func (f *failingVideoProvider) Name() string  { return f.name }
func (f *failingVideoProvider) Model() string { return f.model }
func (f *failingVideoProvider) Generate(_ context.Context, _ VideoRequest) (*VideoResult, error) {
	return nil, fmt.Errorf("status 422: unsupported params")
}

func TestVideoRouter_PicksCheapest(t *testing.T) {
	t.Parallel()
	// fal LTX fast is cheaper than gemini at 720p
	fal := &stubVideoProvider{name: "fal", model: "fal-ai/ltx-2.3/text-to-video/fast", cost: 0.24, elapsed: 30 * time.Second}
	gemini := &stubVideoProvider{name: "gemini", model: "gemini-omni-1.1-flash", cost: 1.00, elapsed: 25 * time.Second}

	router, err := NewVideoRouter(
		WithRoute(fal),
		WithRoute(gemini),
		RouteByPrice(1.0),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := router.Generate(context.Background(), VideoRequest{
		Prompt:     "test",
		Duration:   5,
		Resolution: "720p",
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != fal.model {
		t.Fatalf("routed to %q, want fal (cheaper)", res.Model)
	}
}

func TestVideoRouter_PicksFastest(t *testing.T) {
	t.Parallel()
	slow := &stubVideoProvider{name: "fal", model: "fal-ai/ltx-2.3/text-to-video/fast", elapsed: 40 * time.Second}
	fast := &stubVideoProvider{name: "gemini", model: "gemini-omni-1.1-flash", elapsed: 10 * time.Second}

	router, err := NewVideoRouter(
		WithRoute(slow),
		WithRoute(fast),
		RouteByAvailability(1.0),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Seed latency history
	for range 3 {
		_, _ = router.Generate(context.Background(), VideoRequest{Prompt: "warmup"})
	}

	res, err := router.Generate(context.Background(), VideoRequest{Prompt: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != fast.model {
		t.Fatalf("routed to %q, want gemini (faster)", res.Model)
	}
}

func TestVideoRouter_FiltersNonVideoModel(t *testing.T) {
	t.Parallel()
	// "not-a-video-model" has no video pricing → filtered out
	bad := &stubVideoProvider{name: "bad", model: "not-a-video-model", cost: 0.01}
	good := &stubVideoProvider{name: "gemini", model: "gemini-omni-1.1-flash", cost: 1.00}

	router, err := NewVideoRouter(
		WithRoute(bad),
		WithRoute(good),
		RouteByPrice(1.0),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := router.Generate(context.Background(), VideoRequest{Prompt: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != good.model {
		t.Fatalf("routed to %q, want gemini (only video-capable)", res.Model)
	}
}

func TestVideoRouter_FallbackOnError(t *testing.T) {
	t.Parallel()
	failing := &failingVideoProvider{name: "fal", model: "fal-ai/ltx-2.3/text-to-video/fast"}
	working := &stubVideoProvider{name: "gemini", model: "gemini-omni-1.1-flash", elapsed: 10 * time.Second}

	router, err := NewVideoRouter(
		WithRoute(failing),
		WithRoute(working),
		RouteByPrice(1.0),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := router.Generate(context.Background(), VideoRequest{
		Prompt:     "test",
		Duration:   5,
		Resolution: "720p",
	})
	if err != nil {
		t.Fatalf("expected fallback, got: %v", err)
	}
	if res.Model != working.model {
		t.Fatalf("got %q, want gemini (fallback after fal error)", res.Model)
	}
}

func TestVideoRouter_NoProviders(t *testing.T) {
	t.Parallel()
	_, err := NewVideoRouter()
	if err == nil {
		t.Fatal("expected error with no providers")
	}
}

func TestVideoRouter_DefaultsToPriceRouting(t *testing.T) {
	t.Parallel()
	fal := &stubVideoProvider{name: "fal", model: "fal-ai/ltx-2.3/text-to-video/fast"}
	gemini := &stubVideoProvider{name: "gemini", model: "gemini-omni-1.1-flash"}

	router, err := NewVideoRouter(
		WithRoute(fal),
		WithRoute(gemini),
	)
	if err != nil {
		t.Fatal(err)
	}
	// No RouteByPrice/RouteByAvailability — defaults to price
	res, err := router.Generate(context.Background(), VideoRequest{
		Prompt:     "test",
		Duration:   5,
		Resolution: "720p",
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != fal.model {
		t.Fatalf("default routing picked %q, want fal (cheaper)", res.Model)
	}
}
