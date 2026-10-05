package ai

import (
	"context"
	"io"
)

// STTProvider transcribes audio into text.
type STTProvider interface {
	Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error)
}

// STTMeterable is implemented by speech-to-text providers that report usage.
type STTMeterable interface {
	SetMeter(MeterHook)
}

// SetSTTMeter attaches hook to p when it reports usage.
func SetSTTMeter(p STTProvider, hook MeterHook) {
	if m, ok := p.(STTMeterable); ok {
		m.SetMeter(hook)
	}
}
