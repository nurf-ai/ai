package ai

import (
	"context"
	"io"
)

// STTProvider transcribes audio into text.
type STTProvider interface {
	Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error)
}
