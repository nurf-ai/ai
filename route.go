package ai

import (
	"context"
	"fmt"
	"math"
)

// Dimension scores providers on one optimization axis.
// Lower scores are better. Route normalizes scores across candidates
// before applying weights, so raw scale does not matter.
type Dimension struct {
	Name      string
	Weight    float64
	CanHandle func(provider string) bool
	Score     func(ctx context.Context, provider string) (float64, error)
}

// PriceDim scores by cost — lower is better.
func PriceDim(weight float64, costFn func(provider string) float64) Dimension {
	return Dimension{
		Name:      "price",
		Weight:    weight,
		CanHandle: func(string) bool { return true },
		Score: func(_ context.Context, p string) (float64, error) {
			return costFn(p), nil
		},
	}
}

// AvailabilityDim scores by estimated latency (seconds) — lower is better.
func AvailabilityDim(weight float64, probeFn func(ctx context.Context, provider string) (float64, error)) Dimension {
	return Dimension{
		Name:      "availability",
		Weight:    weight,
		CanHandle: func(string) bool { return true },
		Score:     probeFn,
	}
}

// CapabilityDim is a filter-only dimension that excludes providers
// that cannot handle the request. It has no score weight.
func CapabilityDim(canHandle func(provider string) bool) Dimension {
	return Dimension{
		Name:      "capability",
		Weight:    0,
		CanHandle: canHandle,
	}
}

// RouteResult holds the routing decision and per-dimension scores.
type RouteResult struct {
	Provider string
	Scores   map[string]float64 // dimension name → normalized [0,1]
	Total    float64
}

// Route picks the best provider from candidates using weighted dimensions.
// Each candidate is filtered by CanHandle, scored per dimension, normalized
// to [0,1] within each dimension, then ranked by weighted sum (lowest wins).
func Route(ctx context.Context, candidates []string, dims ...Dimension) (*RouteResult, error) {
	viable := make([]string, 0, len(candidates))
	for _, c := range candidates {
		ok := true
		for _, d := range dims {
			if d.CanHandle != nil && !d.CanHandle(c) {
				ok = false
				break
			}
		}
		if ok {
			viable = append(viable, c)
		}
	}
	if len(viable) == 0 {
		return nil, fmt.Errorf("route: no viable provider")
	}
	if len(viable) == 1 {
		return &RouteResult{Provider: viable[0]}, nil
	}

	totals := make(map[string]float64, len(viable))
	perDim := make(map[string]map[string]float64, len(viable))
	for _, c := range viable {
		perDim[c] = make(map[string]float64)
	}

	for _, d := range dims {
		if d.Weight == 0 || d.Score == nil {
			continue
		}
		raw := make(map[string]float64, len(viable))
		for _, c := range viable {
			s, err := d.Score(ctx, c)
			if err != nil {
				s = math.MaxFloat64
			}
			raw[c] = s
		}
		lo, hi := boundsOf(raw)
		for c, s := range raw {
			var norm float64
			if hi > lo {
				norm = (s - lo) / (hi - lo)
			}
			perDim[c][d.Name] = norm
			totals[c] += norm * d.Weight
		}
	}

	best := viable[0]
	for _, c := range viable[1:] {
		if totals[c] < totals[best] {
			best = c
		}
	}
	return &RouteResult{Provider: best, Scores: perDim[best], Total: totals[best]}, nil
}

func boundsOf(m map[string]float64) (lo, hi float64) {
	lo, hi = math.MaxFloat64, -math.MaxFloat64
	for _, v := range m {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return
}
