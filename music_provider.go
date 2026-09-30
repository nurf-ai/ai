package ai

import (
	"context"
	"time"
)

// MusicRequest describes a single song or instrumental track generation.
type MusicRequest struct {
	// Prompt describes the music: genre, mood, instrumentation, tempo, voice.
	Prompt string
	// Lyrics are the words to sing. Section tags such as [Verse] and [Chorus]
	// are understood by most models. Empty lets the model write its own,
	// unless Instrumental is set.
	Lyrics string
	// Instrumental asks for a track without vocals.
	Instrumental bool
	// Model overrides the provider's default endpoint for this call.
	Model string
}

// MusicResult is the provider-agnostic outcome of a Generate call.
type MusicResult struct {
	URL         string
	ContentType string
	FileName    string
	FileSize    int64
	Duration    float64 // seconds; 0 when the model does not report it
	Lyrics      string  // the words as sung, when the model returns them
	Model       string
	CostUSD     float64
	Elapsed     time.Duration
}

// MusicProvider generates songs and instrumental tracks from a text prompt.
type MusicProvider interface {
	Name() string
	Model() string
	Generate(ctx context.Context, req MusicRequest) (*MusicResult, error)
}

// MusicMeterable is implemented by music providers that accept a meter hook.
type MusicMeterable interface {
	SetMeter(MeterHook)
}

// SetMusicMeter attaches a meter hook to any MusicProvider that satisfies
// MusicMeterable.
func SetMusicMeter(p MusicProvider, hook MeterHook) {
	if hook == nil || p == nil {
		return
	}
	if m, ok := p.(MusicMeterable); ok {
		m.SetMeter(hook)
	}
}
