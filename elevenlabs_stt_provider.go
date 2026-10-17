package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/textproto"
	"path/filepath"

	"go.uber.org/zap"
)

const DefaultElevenLabsSTTModel = "scribe_v2"

// ElevenLabsSTTProvider transcribes audio with POST /v1/speech-to-text
// (Scribe), metered from the audio duration the API reports.
type ElevenLabsSTTProvider struct {
	client *elevenLabsClient
	model  string
	meter  MeterHook
}

func NewElevenLabsSTTProvider(apiKey, model string) *ElevenLabsSTTProvider {
	if model == "" {
		model = DefaultElevenLabsSTTModel
	}
	return &ElevenLabsSTTProvider{client: newElevenLabsClient(apiKey), model: model}
}

// SetMeter attaches the usage hook: one event per transcription, priced by
// the seconds of audio the API reports.
func (p *ElevenLabsSTTProvider) SetMeter(hook MeterHook) { p.meter = hook }

type elevenLabsSTTResponse struct {
	Text              string  `json:"text"`
	LanguageCode      string  `json:"language_code"`
	AudioDurationSecs float64 `json:"audio_duration_secs"`
}

func (p *ElevenLabsSTTProvider) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	logger.Debug("stt transcribe", zap.String("provider", "elevenlabs"), zap.String("model", p.model), zap.String("filename", filename))
	if audio == nil {
		return "", errors.New("elevenlabs stt: no audio")
	}
	body, contentType, size, err := p.form(audio, filename, STTLanguageFromCtx(ctx))
	if err != nil {
		return "", fmt.Errorf("elevenlabs stt: %w", err)
	}
	resp, err := p.client.post(ctx, "/v1/speech-to-text", body, contentType)
	if err != nil {
		return "", fmt.Errorf("elevenlabs stt: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck
	var res elevenLabsSTTResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("elevenlabs stt: decode: %w", err)
	}
	p.emitUsage(ctx, &res, size, filename)
	return res.Text, nil
}

func (p *ElevenLabsSTTProvider) form(audio io.Reader, filename, language string) (*bytes.Buffer, string, int64, error) {
	if filename == "" {
		if f, ok := audio.(interface{ Name() string }); ok {
			filename = f.Name()
		}
	}
	if filename == "" {
		filename = "audio"
	}
	fileType := mime.TypeByExtension(filepath.Ext(filename))
	if fileType == "" {
		fileType = "application/octet-stream"
	}
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	if err := w.WriteField("model_id", p.model); err != nil {
		return nil, "", 0, err
	}
	if language != "" {
		if err := w.WriteField("language_code", language); err != nil {
			return nil, "", 0, err
		}
	}
	part, err := w.CreatePart(textproto.MIMEHeader{
		"Content-Disposition": {fmt.Sprintf(`form-data; name="file"; filename="%s"`, formQuoteEscaper.Replace(filepath.Base(filename)))},
		"Content-Type":        {fileType},
	})
	if err != nil {
		return nil, "", 0, err
	}
	size, err := io.Copy(part, audio)
	if err != nil {
		return nil, "", size, fmt.Errorf("read audio: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, "", size, err
	}
	return body, w.FormDataContentType(), size, nil
}

func (p *ElevenLabsSTTProvider) emitUsage(ctx context.Context, res *elevenLabsSTTResponse, size int64, filename string) {
	if p.meter == nil {
		return
	}
	md := map[string]any{}
	seconds, from := res.AudioDurationSecs, "duration"
	if seconds <= 0 {
		seconds, from = estimateAudioSeconds(size, filename), "size"
		md["estimated"] = true
	}
	md["seconds"], md["seconds_from"] = seconds, from
	if res.LanguageCode != "" {
		md["language"] = res.LanguageCode
	}
	p.meter(UsageEvent{
		CallerID:         MeterCallerIDFromCtx(ctx),
		Provider:         "elevenlabs",
		Model:            p.model,
		Operation:        "stt",
		DebugSpanID:      DebugSpanIDFromCtx(ctx),
		EstimatedCostUSD: EstimateSTTCost(p.model, seconds),
		Metadata:         mergeMeterMetadata(ctx, md),
	})
}
