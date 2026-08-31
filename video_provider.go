package ai

import (
	"context"
	"encoding/base64"
	"time"
)

// VideoRequest describes a single clip generation.
//
// Image / ImageURL are optional first-frame conditioning (image-to-video).
// Leave both empty for text-to-video. Zero-valued knobs mean "provider
// default" — providers only forward fields the caller set.
type VideoRequest struct {
	Prompt         string
	NegativePrompt string
	// ImageURL is forwarded verbatim (https:// or data: URI).
	ImageURL string
	// Image is raw image bytes; encoded as a data: URI when ImageURL is empty.
	Image []byte
	// ImageMediaType is the MIME type of Image ("image/jpeg" when empty).
	ImageMediaType string
	// Duration in seconds. Providers snap to their supported set.
	Duration float64
	// Resolution label, e.g. "1080p" | "1440p" | "2160p".
	Resolution string
	// AspectRatio label, e.g. "auto" | "16:9" | "9:16".
	AspectRatio string
	// FPS of the generated clip; 0 = provider default.
	FPS int
	// Audio asks the model to generate a soundtrack when it can.
	Audio bool
	// Seed pins generation; nil = random per call.
	Seed *int64
	// Model overrides the provider's default model/endpoint for this call.
	Model string
}

// VideoResult is the provider-agnostic outcome of a Generate call.
//
// URL points at a provider-hosted file that is typically temporary — callers
// that need durability must download it promptly.
type VideoResult struct {
	URL         string
	ContentType string
	FileName    string
	FileSize    int64
	Width       int
	Height      int
	FPS         float64
	Duration    float64 // seconds
	NumFrames   int
	Model       string
	Seed        int64
	CostUSD     float64
	Elapsed     time.Duration
}

// VideoProvider generates short video clips from a prompt and an optional
// conditioning image.
type VideoProvider interface {
	Name() string
	Model() string
	Generate(ctx context.Context, req VideoRequest) (*VideoResult, error)
}

// VideoMeterable is implemented by video providers that accept a meter hook.
type VideoMeterable interface {
	SetMeter(MeterHook)
}

// SetVideoMeter attaches a meter hook to any VideoProvider that satisfies
// VideoMeterable.
func SetVideoMeter(v VideoProvider, hook MeterHook) {
	if hook == nil || v == nil {
		return
	}
	if m, ok := v.(VideoMeterable); ok {
		m.SetMeter(hook)
	}
}

// SetVideoModeration attaches a moderation provider to any VideoProvider
// that satisfies the moderable interface.
func SetVideoModeration(v VideoProvider, m ModerationProvider) {
	if m == nil || v == nil {
		return
	}
	type moderable interface{ SetModeration(ModerationProvider) }
	if p, ok := v.(moderable); ok {
		p.SetModeration(m)
	}
}

// DataURI encodes raw bytes as a base64 data: URI.
func DataURI(mediaType string, data []byte) string {
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
}
