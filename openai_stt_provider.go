package ai

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

type OpenAISTTProvider struct {
	raw   *openai.Client
	model string
	meter MeterHook
}

func newOpenAISTTProvider(apiKey, model string) *OpenAISTTProvider {
	if model == "" {
		model = openai.Whisper1
	}
	return &OpenAISTTProvider{raw: openai.NewClient(apiKey), model: model}
}

// SetMeter attaches the usage hook: one event per transcription, priced by
// the audio's duration.
func (p *OpenAISTTProvider) SetMeter(hook MeterHook) { p.meter = hook }

func (p *OpenAISTTProvider) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	logger.Debug("stt transcribe", zap.String("model", p.model), zap.String("filename", filename))
	counted := &countingReader{r: audio}
	req := openai.AudioRequest{
		Model:    p.model,
		Reader:   counted,
		FilePath: filename,
	}
	if p.model == openai.Whisper1 {
		req.Format = openai.AudioResponseFormatVerboseJSON // reports the audio's duration
	}
	resp, err := p.raw.CreateTranscription(ctx, req)
	if err != nil {
		return "", fmt.Errorf("openai stt: %w", err)
	}
	p.emitUsage(ctx, resp.Duration, counted.n, filename)
	return resp.Text, nil
}

func (p *OpenAISTTProvider) emitUsage(ctx context.Context, seconds float64, bytes int64, filename string) {
	if p.meter == nil {
		return
	}
	md := map[string]any{}
	if seconds <= 0 {
		seconds = estimateAudioSeconds(bytes, filename)
		md["estimated"] = true
	}
	md["seconds"] = seconds
	ev := UsageEvent{
		CallerID:         MeterCallerIDFromCtx(ctx),
		Provider:         "openai",
		Model:            p.model,
		Operation:        "stt",
		EstimatedCostUSD: EstimateSTTCost(p.model, seconds),
		DebugSpanID:      DebugSpanIDFromCtx(ctx),
	}
	ev.Metadata = mergeMeterMetadata(ctx, md)
	p.meter(ev)
}

// estimateAudioSeconds guesses a recording's length from its size when the
// API reports no duration: what browsers record (Opus in WebM or Ogg) runs
// near 32 kbps, compressed music formats near 128 kbps, WAV is 16 kHz mono PCM.
func estimateAudioSeconds(bytes int64, filename string) float64 {
	rate := 16000.0 // bytes per second, ~128 kbps (mp3, m4a, mp4)
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".webm", ".ogg", ".oga", ".opus":
		rate = 4000
	case ".wav":
		rate = 32000
	case ".flac":
		rate = 24000
	}
	return float64(bytes) / rate
}

// countingReader counts the bytes read through it.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(b []byte) (int, error) {
	n, err := c.r.Read(b)
	c.n += int64(n)
	return n, err
}
