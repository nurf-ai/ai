package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// UsageEvent represents a single LLM/image-gen usage event for metering.
type UsageEvent struct {
	CallerID           uuid.UUID `json:"caller_id"`
	Provider         string    `json:"provider"`
	Model            string    `json:"model"`
	Operation        string    `json:"operation"`
	InputTokens      int       `json:"input_tokens"`
	OutputTokens     int       `json:"output_tokens"`
	TotalTokens      int       `json:"total_tokens"`
	EstimatedCostUSD float64   `json:"estimated_cost_usd"`
	// CacheCreationInputTokens counts tokens written to the provider's
	// prompt cache on this call (Anthropic only, populated when cache
	// writes occur; priced at ~1.25x base). Zero on cache hit or when
	// caching isn't active.
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	// CacheReadInputTokens counts tokens served from the provider's
	// prompt cache (Anthropic only; priced at ~0.1x base). Nonzero means
	// WithCacheSysPrompt is actively paying off.
	CacheReadInputTokens int `json:"cache_read_input_tokens,omitempty"`
	// SystemPrompt/UserPrompt are captured for dev debug display only — empty
	// in production (see capturePromptForDebug).
	SystemPrompt string `json:"system_prompt,omitempty"`
	UserPrompt   string `json:"user_prompt,omitempty"`
	// DebugSpanID is the ID of the PromptMiddlewareChain span that was active
	// when this LLM call fired — used by the debug panel meter hook to attach
	// the llm_call event to the correct span in the tree. Empty in prod.
	DebugSpanID string         `json:"debug_span_id,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// debugPromptsEnabled controls whether LLM providers capture system/user
// prompts into UsageEvents for the dev debug panel. Disabled by default — set
// to true at startup in dev mode via SetDebugPromptsEnabled. When false,
// providers skip the prompt capture entirely (no allocation) so production
// doesn't pay the cost of building debug strings that nobody will see.
var debugPromptsEnabled bool

// SetDebugPromptsEnabled toggles prompt capture in UsageEvents. Call once at
// startup based on your dev-mode flag.
func SetDebugPromptsEnabled(enabled bool) { debugPromptsEnabled = enabled }

// DebugPromptsEnabled reports whether prompt capture is active. Providers
// check this before building sys/user prompt strings for the debug panel.
func DebugPromptsEnabled() bool { return debugPromptsEnabled }

// capturePromptForDebug returns the full prompt when debug prompts are enabled,
// or the empty string otherwise. Used by all providers when emitting UsageEvents
// so prod skips the string work entirely but dev sees the full untruncated
// prompt that was actually sent to the LLM.
func capturePromptForDebug(s string) string {
	if !debugPromptsEnabled {
		return ""
	}
	return s
}

// TruncatePromptForDebug is kept for callers that still reference the old
// helper name; it now just delegates to capturePromptForDebug — prompts are
// no longer truncated. Dev mode only carries them at all.
func TruncatePromptForDebug(s string) string {
	return capturePromptForDebug(s)
}

// MeterHook is called after each LLM/image call with usage data.
type MeterHook func(UsageEvent)

// BlockSize carries the two attribution signals per prompt block: chars (raw
// length) and tokens (tiktoken o200k_base count). Tokens are what bills you;
// chars exist for sanity (a high tokens/chars ratio per block flags
// tokenizer-hostile content like code, base64, or heavy unicode — surfd
// as the "density" warning in PromptBlocksBar).
type BlockSize struct {
	Chars  int `json:"chars"`
	Tokens int `json:"tokens"`
}

// PromptBlocks is a per-call breakdown of the prompt into named blocks.
// Each entry carries chars + tokens. The meter hook stamps this onto
// UsageEvent.Metadata["blocks"] so downstream recorders can see the breakdown
// without needing each provider to crack the prompt apart.
type PromptBlocks map[string]BlockSize

// Context keys for metering metadata.
type meterCtxKey int

const (
	ctxKeyMeterCallerID meterCtxKey = iota
	ctxKeyMeterOperation
	ctxKeyTransparentBG
	ctxKeyDebugSpanID
	ctxKeyPromptBlocks
	ctxKeyCacheSysPrompt
	ctxKeyMeterMetadata
)

// WithDebugSpanID stamps the current dev-debug span ID on the context so
// providers can include it on UsageEvents. Caller must be in dev mode; in
// prod this is a no-op path (the middleware chain never calls StartSpan).
func WithDebugSpanID(ctx context.Context, spanID string) context.Context {
	return context.WithValue(ctx, ctxKeyDebugSpanID, spanID)
}

// DebugSpanIDFromCtx returns the span ID on the context, or empty.
func DebugSpanIDFromCtx(ctx context.Context) string {
	if id, ok := ctx.Value(ctxKeyDebugSpanID).(string); ok {
		return id
	}
	return ""
}

func WithMeterCallerID(ctx context.Context, id uuid.UUID) context.Context {
	return context.WithValue(ctx, ctxKeyMeterCallerID, id)
}

func WithMeterOperation(ctx context.Context, op string) context.Context {
	return context.WithValue(ctx, ctxKeyMeterOperation, op)
}

func MeterCallerIDFromCtx(ctx context.Context) uuid.UUID {
	if id, ok := ctx.Value(ctxKeyMeterCallerID).(uuid.UUID); ok {
		return id
	}
	return uuid.Nil
}

func MeterOperationFromCtx(ctx context.Context) string {
	if op, ok := ctx.Value(ctxKeyMeterOperation).(string); ok {
		return op
	}
	return "unknown"
}

// WithMeterMetadata stamps arbitrary key/values on the context so every
// provider merges them into UsageEvent.Metadata (surf handle, session id,
// feature flag — whatever the caller wants attributed). Stacking calls merges
// kv over what is already stamped, later keys win, into a fresh map: the
// stored map is copied, never mutated, and kv is copied too so the caller may
// reuse it. Empty/nil kv returns ctx unchanged.
func WithMeterMetadata(ctx context.Context, kv map[string]any) context.Context {
	if len(kv) == 0 {
		return ctx
	}
	prev, _ := ctx.Value(ctxKeyMeterMetadata).(map[string]any)
	merged := make(map[string]any, len(prev)+len(kv))
	maps.Copy(merged, prev)
	maps.Copy(merged, kv)
	return context.WithValue(ctx, ctxKeyMeterMetadata, merged)
}

// MeterMetadataFromCtx returns a copy of the metadata stamped on ctx via
// WithMeterMetadata, or nil when none. Mutating the result never touches
// the context.
func MeterMetadataFromCtx(ctx context.Context) map[string]any {
	md, _ := ctx.Value(ctxKeyMeterMetadata).(map[string]any)
	if len(md) == 0 {
		return nil
	}
	return maps.Clone(md)
}

// mergeMeterMetadata returns md plus every ctx entry whose key md does not
// already carry — provider-set keys win over caller-stamped ones. Returns
// nil when both are empty. The ctx map is never mutated; md may be (it is
// allocated when nil and ctx has entries). Each adapter calls this as the
// very last step before its meter hook — after attachBlocks and any other
// per-provider stamping — so nothing appended later is missed.
func mergeMeterMetadata(ctx context.Context, md map[string]any) map[string]any {
	ctxMD, _ := ctx.Value(ctxKeyMeterMetadata).(map[string]any)
	if len(ctxMD) == 0 {
		if len(md) == 0 {
			return nil
		}
		return md
	}
	if md == nil {
		md = make(map[string]any, len(ctxMD))
	}
	for k, v := range ctxMD {
		if _, ok := md[k]; !ok {
			md[k] = v
		}
	}
	return md
}

// WithPromptBlocks stamps a per-block breakdown on the context so providers
// can include it on UsageEvents. Call-site passes the raw block strings keyed
// by block name — this helper
// computes both Chars (len(s)) and Tokens (CountTokens(s)) per entry.
//
// Tokenizer cost is paid here, but `dev-only` callers and a single tokenize
// per call mean it's negligible vs the LLM round-trip. If you need to skip the
// tokenize (e.g. hot path that doesn't care about tokens), call
// WithPromptBlocksRaw with a pre-built map[string]BlockSize directly.
//
// Empty maps are intentionally not stored — PromptBlocksFromCtx returning nil
// is the canonical "absent" signal.
func WithPromptBlocks(ctx context.Context, contents map[string]string) context.Context {
	if len(contents) == 0 {
		return ctx
	}
	b := make(PromptBlocks, len(contents))
	for name, s := range contents {
		b[name] = BlockSize{Chars: len(s), Tokens: CountTokens(s)}
	}
	return context.WithValue(ctx, ctxKeyPromptBlocks, b)
}

// WithPromptBlocksRaw is the lower-level variant — caller pre-computes Chars
// and Tokens per block. Useful when the same block string is reused across
// calls and you want to cache the token count.
func WithPromptBlocksRaw(ctx context.Context, b PromptBlocks) context.Context {
	if len(b) == 0 {
		return ctx
	}
	return context.WithValue(ctx, ctxKeyPromptBlocks, b)
}

// PromptBlocksFromCtx returns the prompt-block breakdown stamped on ctx, or
// nil if none. Providers call this inside emitUsage to enrich the
// UsageEvent.Metadata.
func PromptBlocksFromCtx(ctx context.Context) PromptBlocks {
	if b, ok := ctx.Value(ctxKeyPromptBlocks).(PromptBlocks); ok {
		return b
	}
	return nil
}

// WithCacheSysPrompt signals that the sys prompt sent in this call is a
// stable prefix worth provider-side prompt caching. Anthropic's adapter
// sets cache_control: ephemeral on the sys block; OpenAI/Ollama/HF are
// no-op (OpenAI auto-caches prompts ≥1024 tokens; Ollama uses KV-cache
// for matching prefixes at the inference layer; HF depends on backend).
//
// Call this at the call site right before the LLM invocation. Marker is
// per-call, not global — stamp only when the caller knows the sys
// prompt is stable across turns (e.g. planner/router system prompts).
func WithCacheSysPrompt(ctx context.Context) context.Context {
	return context.WithValue(ctx, ctxKeyCacheSysPrompt, true)
}

// CacheSysPromptFromCtx reports whether WithCacheSysPrompt was stamped.
// Adapters call this while building the request to decide whether to
// mark the sys block as cacheable.
func CacheSysPromptFromCtx(ctx context.Context) bool {
	v, _ := ctx.Value(ctxKeyCacheSysPrompt).(bool)
	return v
}

// WithTransparentBG signals image providers to produce transparent backgrounds.
func WithTransparentBG(ctx context.Context) context.Context {
	return context.WithValue(ctx, ctxKeyTransparentBG, true)
}

func TransparentBGFromCtx(ctx context.Context) bool {
	v, _ := ctx.Value(ctxKeyTransparentBG).(bool)
	return v
}

// attachBlocks enriches a UsageEvent with the PromptBlocks from ctx, under
// metadata key "blocks". No-op if ctx carries no blocks. Called by each
// adapter's emitUsage so per-provider code stays uniform — all the stamping
// logic lives here.
func attachBlocks(ctx context.Context, ev *UsageEvent) {
	b := PromptBlocksFromCtx(ctx)
	if len(b) == 0 {
		return
	}
	if ev.Metadata == nil {
		ev.Metadata = make(map[string]any, 1)
	}
	// Copy into a plain map[string]BlockSize so JSON marshalling on the wire
	// is stable regardless of whether downstream type-asserts PromptBlocks.
	out := make(map[string]BlockSize, len(b))
	maps.Copy(out, b)
	ev.Metadata["blocks"] = out
}

// LLMMeterable is implemented by providers that accept a meter hook.
// SetLLMMeter uses this instead of a type-switch so new providers work
// without updating the switch.
type LLMMeterable interface {
	SetMeter(MeterHook)
}

// SetLLMMeter attaches a meter hook to any LLMProvider that satisfies
// LLMMeterable.
func SetLLMMeter(llm LLMProvider, hook MeterHook) {
	if hook == nil {
		return
	}
	if m, ok := llm.(LLMMeterable); ok {
		m.SetMeter(hook)
	}
}

// ImageMeterable is implemented by image providers that accept a meter hook.
type ImageMeterable interface {
	SetMeter(MeterHook)
}

// SetImageMeter attaches a meter hook to any ImageProvider that satisfies
// ImageMeterable.
func SetImageMeter(img ImageProvider, hook MeterHook) {
	if hook == nil || img == nil {
		return
	}
	if m, ok := img.(ImageMeterable); ok {
		m.SetMeter(hook)
	}
}

// HTTPMeterOpts configures an HTTPMeterEmitter.
type HTTPMeterOpts struct {
	Endpoint      string                             // full URL — no suffix appended
	AuthHeader    string                             // sent as Authorization header when non-empty
	BatchSize     int                                // default 32
	FlushInterval time.Duration                      // default 2s
	Marshal       func([]UsageEvent) ([]byte, error) // default json.Marshal
	ContentType   string                             // default "application/json"
	OnError       func(error)                        // default logger.Warn
}

// HTTPMeterEmitter buffers usage events and POSTs them to a meter service.
type HTTPMeterEmitter struct {
	ch     chan UsageEvent
	opts   HTTPMeterOpts
	client *http.Client
}

func NewHTTPMeterEmitter(opts HTTPMeterOpts) *HTTPMeterEmitter {
	if opts.BatchSize <= 0 {
		opts.BatchSize = 32
	}
	if opts.FlushInterval <= 0 {
		opts.FlushInterval = 2 * time.Second
	}
	if opts.Marshal == nil {
		opts.Marshal = func(events []UsageEvent) ([]byte, error) {
			return json.Marshal(events)
		}
	}
	if opts.ContentType == "" {
		opts.ContentType = "application/json"
	}
	if opts.OnError == nil {
		opts.OnError = func(err error) {
			logger.Warn("flush error", zap.Error(err))
		}
	}
	e := &HTTPMeterEmitter{
		ch:     make(chan UsageEvent, 1024),
		opts:   opts,
		client: &http.Client{Timeout: 5 * time.Second},
	}
	go e.run()
	return e
}

func (e *HTTPMeterEmitter) Hook() MeterHook {
	return func(ev UsageEvent) {
		select {
		case e.ch <- ev:
		default:
			logger.Warn("buffer full, dropping event")
		}
	}
}

func (e *HTTPMeterEmitter) run() {
	batch := make([]UsageEvent, 0, e.opts.BatchSize)
	ticker := time.NewTicker(e.opts.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case ev := <-e.ch:
			batch = append(batch, ev)
			if len(batch) >= e.opts.BatchSize {
				e.flush(batch)
				batch = batch[:0]
			}
		case <-ticker.C:
			if len(batch) > 0 {
				e.flush(batch)
				batch = batch[:0]
			}
		}
	}
}

func (e *HTTPMeterEmitter) flush(batch []UsageEvent) {
	body, err := e.opts.Marshal(batch)
	if err != nil {
		e.opts.OnError(fmt.Errorf("marshal: %w", err))
		return
	}
	req, err := http.NewRequest(http.MethodPost, e.opts.Endpoint, bytes.NewReader(body))
	if err != nil {
		e.opts.OnError(fmt.Errorf("build request: %w", err))
		return
	}
	req.Header.Set("Content-Type", e.opts.ContentType)
	if e.opts.AuthHeader != "" {
		req.Header.Set("Authorization", e.opts.AuthHeader)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		e.opts.OnError(fmt.Errorf("POST %s: %w", e.opts.Endpoint, err))
		return
	}
	_ = resp.Body.Close()
	if resp.StatusCode >= 300 {
		e.opts.OnError(fmt.Errorf("POST %s: status %d (%d events)", e.opts.Endpoint, resp.StatusCode, len(batch)))
	}
}
