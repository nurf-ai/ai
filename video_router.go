package ai

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"go.uber.org/zap"
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
// If the chosen provider fails, it falls back to the next best.
// Generate tries providers in order and returns the first success.
//
// A request that names a model goes to that model's provider first (the
// caller chose it for a reason — price, native audio, …); only if that fails
// (balance exhausted, outage, moderation) do the others get a turn, each on
// its own default model since model ids are provider-specific. A request
// without a model follows the price/availability ranking as before.
func (r *VideoRouter) Generate(ctx context.Context, req VideoRequest) (*VideoResult, error) {
	dims := r.buildDims(req)

	ranked, err := RouteAll(ctx, r.order, dims...)
	if err != nil {
		return nil, fmt.Errorf("video router: %w", err)
	}

	owner := r.ownerOf(req.Model)
	order := make([]string, 0, len(ranked)+1)
	if owner != "" {
		order = append(order, owner)
	}
	for _, c := range ranked {
		if c.Provider != owner {
			order = append(order, c.Provider)
		}
	}

	var lastErr error
	for _, name := range order {
		provider := r.providers[name]
		pr := req
		if name != owner {
			pr.Model = "" // not this provider's id → its default model
		}
		res, err := provider.Generate(ctx, pr)
		if err != nil {
			logger.Warn("video router: provider failed, trying next",
				zap.String("provider", name), zap.String("model", provider.Model()), zap.Error(err))
			lastErr = err
			continue
		}

		r.mu.Lock()
		r.latency[name].record(res.Elapsed.Seconds())
		r.mu.Unlock()

		return res, nil
	}
	return nil, fmt.Errorf("video router: all providers failed (last: %w)", lastErr)
}

// ownerOf names the routed provider that serves model, or "" when the model
// is empty/unknown. Matches by exact default model, then by models.json
// group (fal-ai/… → "fal"; a veo-* id under the gemini group → "veo").
func (r *VideoRouter) ownerOf(model string) string {
	if model == "" {
		return ""
	}
	for _, name := range r.order {
		if r.providers[name].Model() == model {
			return name
		}
	}
	company := ModelCompany(model)
	if company == "gemini" && strings.HasPrefix(model, "veo") {
		company = "veo"
	}
	if _, ok := r.providers[company]; ok {
		return company
	}
	return ""
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
			cost := EstimateVideoCost(model, dur, res)
			if cost <= 0 {
				return math.MaxFloat64
			}
			return cost
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
		return IsVideoModel(r.providers[p].Model())
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
