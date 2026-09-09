package ai

import (
	"context"
	"time"
)

// AudioRequest describes a single sound-effect generation.
type AudioRequest struct {
	Prompt string
	// Duration in seconds; 0 = provider default.
	Duration int
	// Format is the audio container: wav, mp3, aac (default), flac.
	Format string
}

// AudioResult is the provider-agnostic outcome of a Generate call.
type AudioResult struct {
	URL         string
	ContentType string
	FileName    string
	FileSize    int64
	Duration    float64 // seconds actually generated
	Model       string
	CostUSD     float64
	Elapsed     time.Duration
}

// AudioProvider generates sound effects from a text prompt.
type AudioProvider interface {
	Name() string
	Model() string
	Generate(ctx context.Context, req AudioRequest) (*AudioResult, error)
}

// AudioMeterable is implemented by audio providers that accept a meter hook.
type AudioMeterable interface {
	SetMeter(MeterHook)
}

// SetAudioMeter attaches a meter hook to any AudioProvider that satisfies
// AudioMeterable.
func SetAudioMeter(a AudioProvider, hook MeterHook) {
	if hook == nil || a == nil {
		return
	}
	if m, ok := a.(AudioMeterable); ok {
		m.SetMeter(hook)
	}
}
