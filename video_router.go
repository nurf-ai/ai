package ai

import (
	"context"
	"fmt"
	"sync"
)

// VideoRouter routes Generate calls across multiple VideoProviders using
// weighted Dimensions (price, availability, capability). It implements
// VideoProvider so callers can use it as a drop-in replacement.
type VideoRouter struct {
	providers map[string]VideoProvider
	order     []string
	pricew    float64
	availw    float64
	latency   map[string]*latencyTracker
	mu        sync.RWMutex
}

// VideoRouterOption configures a VideoRouter.
type VideoRouterOption func(*VideoRouter)

// NewVideoRouter creates a router that picks the best provider per request.
// Without explicit dimension options it defaults to price-only routing.
func NewVideoRouter(opts ...VideoRouterOption) (*VideoRouter, error) {
	r := &VideoRouter{
		providers: make(map[string]VideoProvider),
		latency:   make(map[string]*latencyTracker),
	}
	for _, o := range opts {
		o(r)
	}
	if len(r.providers) == 0 {
		return nil, fmt.Errorf("video router: no providers")
	}
	if r.pricew == 0 && r.availw == 0 {
		r.pricew = 1.0
	}
	return r, nil
}

// WithRoute adds a VideoProvider to the router.
func WithRoute(p VideoProvider) VideoRouterOption {
	return func(r *VideoRouter) {
		name := p.Name()
		r.providers[name] = p
		r.order = append(r.order, name)
		r.latency[name] = &latencyTracker{samples: make([]float64, 0, latencyWindow)}
	}
}

// RouteByPrice enables price-weighted scoring.
func RouteByPrice(weight float64) VideoRouterOption {
	return func(r *VideoRouter) { r.pricew = weight }
}

// RouteByAvailability enables availability-weighted scoring based on
// historical latency per provider.
func RouteByAvailability(weight float64) VideoRouterOption {
	return func(r *VideoRouter) { r.availw = weight }
}

func (r *VideoRouter) Name() string  { return "router" }
func (r *VideoRouter) Model() string { return "" }

// SetMeter forwards the meter hook to all underlying providers.
func (r *VideoRouter) SetMeter(hook MeterHook) {
	for _, p := range r.providers {
		if m, ok := p.(VideoMeterable); ok {
			m.SetMeter(hook)
		}
	}
}

// Generate picks the best provider for req and delegates to it.
func (r *VideoRouter) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
	dims := r.buildDims(req)

	result, err := Route(ctx, r.order, dims...)
	if err != nil {
		return nil, fmt.Errorf("video router: %w", err)
	}

	provider := r.providers[result.Provider]
	res, err := provider.Generate(ctx, req)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	r.latency[result.Provider].record(res.Elapsed.Seconds())
	r.mu.Unlock()

	return res, nil
}

func (r *VideoRouter) buildDims(req VideoRequest) []Dimension {
	var dims []Dimension

	if r.pricew > 0 {
		dims = append(dims, PriceDim(r.pricew, func(p string) float64 {
			model := r.providers[p].Model()
			dur := req.Duration
			if dur == 0 {
				dur = 10
			}
			res := req.Resolution
			if res == "" {
				res = "720p"
			}
			return EstimateVideoCost(model, dur, res)
		}))
	}

	if r.availw > 0 {
		r.mu.RLock()
		defer r.mu.RUnlock()
		dims = append(dims, AvailabilityDim(r.availw, func(_ context.Context, p string) (float64, error) {
			return r.latency[p].average(), nil
		}))
	}

	dims = append(dims, CapabilityDim(func(p string) bool {
		model := r.providers[p].Model()
		if !IsVideoModel(model) {
			return false
		}
		res := req.Resolution
		if res == "" {
			return true
		}
		table := PricingTable()
		mp, ok := table[model]
		if !ok || mp.PerVideoSecondByResolution == nil {
			return true
		}
		_, supported := mp.PerVideoSecondByResolution[res]
		return supported
	}))

	return dims
}

const latencyWindow = 10

type latencyTracker struct {
	samples []float64
	pos     int
	full    bool
}

func (t *latencyTracker) record(seconds float64) {
	if len(t.samples) < latencyWindow {
		t.samples = append(t.samples, seconds)
	} else {
		t.samples[t.pos] = seconds
	}
	t.pos = (t.pos + 1) % latencyWindow
	if t.pos == 0 {
		t.full = true
	}
}

func (t *latencyTracker) average() float64 {
	n := len(t.samples)
	if n == 0 {
		return 0
	}
	var sum float64
	for _, s := range t.samples {
		sum += s
	}
	return sum / float64(n)
}
