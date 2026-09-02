package ai

import (
	"context"
	"math"
	"testing"
)

func TestRoute_PriceOnly(t *testing.T) {
	t.Parallel()
	r, err := Route(context.Background(), []string{"expensive", "cheap"},
		PriceDim(1.0, func(p string) float64 {
			if p == "cheap" {
				return 0.10
			}
			return 1.00
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "cheap" {
		t.Fatalf("got %q, want cheap", r.Provider)
	}
}

func TestRoute_AvailabilityOnly(t *testing.T) {
	t.Parallel()
	r, err := Route(context.Background(), []string{"slow", "fast"},
		AvailabilityDim(1.0, func(_ context.Context, p string) (float64, error) {
			if p == "fast" {
				return 2.0, nil
			}
			return 30.0, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "fast" {
		t.Fatalf("got %q, want fast", r.Provider)
	}
}

func TestRoute_WeightedPriceAndAvailability(t *testing.T) {
	t.Parallel()
	// "cheap" is cheaper but slower, "fast" is expensive but fast.
	// With heavy availability weight, "fast" should win.
	r, err := Route(context.Background(), []string{"cheap", "fast"},
		PriceDim(0.2, func(p string) float64 {
			if p == "cheap" {
				return 0.10
			}
			return 1.00
		}),
		AvailabilityDim(0.8, func(_ context.Context, p string) (float64, error) {
			if p == "fast" {
				return 1.0, nil
			}
			return 60.0, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "fast" {
		t.Fatalf("got %q, want fast (availability weighted 0.8)", r.Provider)
	}
}

func TestRoute_CapabilityFilters(t *testing.T) {
	t.Parallel()
	r, err := Route(context.Background(), []string{"no4k", "has4k"},
		CapabilityDim(func(p string) bool { return p == "has4k" }),
		PriceDim(1.0, func(p string) float64 {
			if p == "no4k" {
				return 0.01
			}
			return 10.0
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "has4k" {
		t.Fatalf("got %q, want has4k (only viable)", r.Provider)
	}
}

func TestRoute_AllFilteredOut(t *testing.T) {
	t.Parallel()
	_, err := Route(context.Background(), []string{"a", "b"},
		CapabilityDim(func(string) bool { return false }),
	)
	if err == nil {
		t.Fatal("expected error when all filtered")
	}
}

func TestRoute_SingleCandidate(t *testing.T) {
	t.Parallel()
	r, err := Route(context.Background(), []string{"only"},
		PriceDim(1.0, func(string) float64 { return 5.0 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "only" {
		t.Fatalf("got %q, want only", r.Provider)
	}
}

func TestRoute_EqualScores(t *testing.T) {
	t.Parallel()
	r, err := Route(context.Background(), []string{"a", "b"},
		PriceDim(1.0, func(string) float64 { return 0.50 }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != "a" && r.Provider != "b" {
		t.Fatalf("got %q, want a or b", r.Provider)
	}
}

func TestLatencyTracker(t *testing.T) {
	t.Parallel()
	tr := &latencyTracker{samples: make([]float64, 0, latencyWindow)}

	if avg := tr.average(); avg != 0 {
		t.Fatalf("empty average = %f, want 0", avg)
	}

	tr.record(10.0)
	tr.record(20.0)
	if avg := tr.average(); math.Abs(avg-15.0) > 0.01 {
		t.Fatalf("average = %f, want 15.0", avg)
	}

	for i := range latencyWindow {
		tr.record(float64(i + 1))
	}
	// window is full, oldest samples evicted
	var sum float64
	for i := range latencyWindow {
		sum += float64(i + 1)
	}
	want := sum / float64(latencyWindow)
	if avg := tr.average(); math.Abs(avg-want) > 0.01 {
		t.Fatalf("average = %f, want %f", avg, want)
	}
}
