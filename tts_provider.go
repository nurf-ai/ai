package ai

import (
	"context"
	"io"
)

// TTSRequest is one text-to-speech synthesis call.
type TTSRequest struct {
	Text string
	// Voice is a provider voice id; empty picks the provider default.
	Voice string
	// Format is the audio container: mp3 (default), opus, aac, flac, wav, pcm.
	Format string
	// Speed scales playback, 0.25–4.0; 0 means 1.0.
	Speed float64
	// Instructions steer delivery (tone, accent, pacing) on models that
	// accept them; ignored elsewhere.
	Instructions string
}

// TTSProvider turns text into speech audio. Synthesize returns the encoded
// audio as a stream the caller must close; bytes arrive progressively on
// providers that stream, so a reader can start playback before the call
// finishes.
type TTSProvider interface {
	Synthesize(ctx context.Context, req TTSRequest) (io.ReadCloser, error)
}

// TTSMeterable is implemented by TTS providers that accept a meter hook.
type TTSMeterable interface {
	SetMeter(MeterHook)
}

// SetTTSMeter attaches a meter hook to any TTSProvider that satisfies
// TTSMeterable.
func SetTTSMeter(t TTSProvider, hook MeterHook) {
	if hook == nil || t == nil {
		return
	}
	if m, ok := t.(TTSMeterable); ok {
		m.SetMeter(hook)
	}
}

// TTSMimeType returns the MIME type for a TTSRequest.Format value.
func TTSMimeType(format string) string {
	switch format {
	case "", "mp3":
		return "audio/mpeg"
	case "opus":
		return "audio/ogg"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm"
	default:
		return "application/octet-stream"
	}
}
