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
	"net/http"
	"net/textproto"
	"path/filepath"
	"strings"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

// sttAudioTokensPerSecond turns the audio tokens a transcription bills into
// seconds of audio. OpenAI: "Audio tokens in user messages are 1 token per
// 100 ms of audio" (https://developers.openai.com/api/docs/guides/realtime-costs).
// Counts include a few special tokens, so the seconds run slightly long.
const sttAudioTokensPerSecond = 10.0

// OpenAISTTProvider transcribes audio with POST /audio/transcriptions. It
// sends the request itself: go-openai's response type drops the usage the
// API reports, and the usage is what a transcription is billed on.
type OpenAISTTProvider struct {
	apiKey  string
	baseURL string
	client  openai.HTTPDoer
	model   string
	meter   MeterHook
}

func newOpenAISTTProvider(apiKey, model string) *OpenAISTTProvider {
	if model == "" {
		model = openai.Whisper1
	}
	cfg := openai.DefaultConfig(apiKey)
	return &OpenAISTTProvider{apiKey: apiKey, baseURL: cfg.BaseURL, client: cfg.HTTPClient, model: model}
}

// SetMeter attaches the usage hook: one event per transcription, priced from
// the usage the API reports (tokens, or seconds of audio).
func (p *OpenAISTTProvider) SetMeter(hook MeterHook) { p.meter = hook }

func (p *OpenAISTTProvider) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	logger.Debug("stt transcribe", zap.String("model", p.model), zap.String("filename", filename))
	if audio == nil {
		return "", errors.New("openai stt: no audio")
	}
	res, size, err := p.transcribe(ctx, audio, filename)
	if err != nil {
		return "", fmt.Errorf("openai stt: %w", err)
	}
	p.emitUsage(ctx, res, size, filename)
	return res.Text, nil
}

// sttResponse is the json answer of POST /audio/transcriptions (verbose_json
// for whisper-1), as far as this provider reads it. Shapes:
// https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
type sttResponse struct {
	Text string `json:"text"`
	// Duration is the audio's length in seconds (verbose_json only).
	Duration float64 `json:"duration"`
	// Usage stays raw so that a shape this code does not know costs the
	// metering, never the transcript. See sttUsage.
	Usage json.RawMessage `json:"usage"`
}

// sttUsage reads both variants of a transcription's usage object:
// {"type":"tokens","input_tokens","input_token_details":{"audio_tokens",
// "text_tokens"},"output_tokens","total_tokens"} from models billed by token
// (gpt-4o-transcribe, gpt-4o-mini-transcribe), and {"type":"duration",
// "seconds"} from models billed by audio length (whisper-1).
type sttUsage struct {
	Seconds           float64 `json:"seconds"`
	InputTokens       int     `json:"input_tokens"`
	OutputTokens      int     `json:"output_tokens"`
	TotalTokens       int     `json:"total_tokens"`
	InputTokenDetails *struct {
		AudioTokens int `json:"audio_tokens"`
		TextTokens  int `json:"text_tokens"`
	} `json:"input_token_details"`
}

// usage is the answer's usage, zero when there is none or it does not parse.
func (r *sttResponse) usage() sttUsage {
	var u sttUsage
	if len(r.Usage) > 0 && json.Unmarshal(r.Usage, &u) != nil {
		logger.Warn("openai stt: unreadable usage, metered without it", zap.ByteString("usage", r.Usage))
		return sttUsage{}
	}
	return u
}

// billedByToken reports the tokens variant.
func (u sttUsage) billedByToken() bool { return u.InputTokens > 0 || u.OutputTokens > 0 }

// inputSplit is the input tokens by kind. Without a breakdown every input
// token is audio: this provider sends no prompt text.
func (u sttUsage) inputSplit() (audio, text int) {
	if d := u.InputTokenDetails; d != nil {
		return d.AudioTokens, d.TextTokens
	}
	return u.InputTokens, 0
}

// transcribe uploads audio and decodes the answer; size is the bytes sent.
func (p *OpenAISTTProvider) transcribe(ctx context.Context, audio io.Reader, filename string) (*sttResponse, int64, error) {
	body, contentType, size, err := sttForm(audio, filename, p.model, openAISTTLanguage(ctx))
	if err != nil {
		return nil, size, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.baseURL, "/")+"/audio/transcriptions", body)
	if err != nil {
		return nil, size, err
	}
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("Accept", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, size, err
	}
	defer resp.Body.Close() //nolint:errcheck
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusBadRequest {
		return nil, size, openAIErrorResponse(resp)
	}
	var res sttResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, size, fmt.Errorf("decode transcription: %w", err)
	}
	return &res, size, nil
}

// openAISTTLanguage is the language hint on ctx in the form the endpoint
// takes: an ISO-639-1 code, so a 3 letter one is left out.
func openAISTTLanguage(ctx context.Context) string {
	if l := STTLanguageFromCtx(ctx); len(l) == 2 {
		return l
	}
	return ""
}

var formQuoteEscaper = strings.NewReplacer(`\`, `\\`, `"`, `\"`)

// sttForm builds the multipart body the endpoint takes, as go-openai did: the
// audio as "file" under its name (the API reads the format from the
// extension), the model, and for whisper-1 response_format=verbose_json,
// which adds the audio's duration. The other models answer json only. A
// non-empty language is sent as the language parameter.
func sttForm(audio io.Reader, filename, model, language string) (body *bytes.Buffer, contentType string, size int64, err error) {
	if filename == "" {
		if f, ok := audio.(interface{ Name() string }); ok {
			filename = f.Name()
		}
	}
	fileType := mime.TypeByExtension(filepath.Ext(filename))
	if fileType == "" {
		fileType = "application/octet-stream"
	}
	body = &bytes.Buffer{}
	w := multipart.NewWriter(body)
	part, err := w.CreatePart(textproto.MIMEHeader{
		"Content-Disposition": {fmt.Sprintf(`form-data; name="file"; filename="%s"`, formQuoteEscaper.Replace(filepath.Base(filename)))},
		"Content-Type":        {fileType},
	})
	if err != nil {
		return nil, "", 0, err
	}
	if size, err = io.Copy(part, audio); err != nil {
		return nil, "", size, fmt.Errorf("read audio: %w", err)
	}
	if err = w.WriteField("model", model); err != nil {
		return nil, "", size, err
	}
	if language != "" {
		if err = w.WriteField("language", language); err != nil {
			return nil, "", size, err
		}
	}
	if model == openai.Whisper1 {
		if err = w.WriteField("response_format", string(openai.AudioResponseFormatVerboseJSON)); err != nil {
			return nil, "", size, err
		}
	}
	if err = w.Close(); err != nil {
		return nil, "", size, err
	}
	return body, w.FormDataContentType(), size, nil
}

// openAIErrorResponse reads a failed answer into the error go-openai returns
// for one, so ClassifyError reads it the same way: *openai.APIError when the
// body carries {"error": …}, *openai.RequestError otherwise.
func openAIErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read error response: %w", err)
	}
	var er openai.ErrorResponse
	if err := json.Unmarshal(body, &er); err != nil || er.Error == nil {
		return &openai.RequestError{HTTPStatus: resp.Status, HTTPStatusCode: resp.StatusCode, Err: err, Body: body}
	}
	er.Error.HTTPStatus = resp.Status
	er.Error.HTTPStatusCode = resp.StatusCode
	return er.Error
}

// emitUsage reports one transcription. Tokens, from a model billed by token,
// are priced per token; otherwise the seconds are priced per minute.
// Metadata["seconds"] is the audio's length and Metadata["seconds_from"] its
// source: "duration" (reported by the API), "audio_tokens" (derived at
// sttAudioTokensPerSecond) or "size" (guessed from the upload, and then
// Metadata["estimated"] is true).
func (p *OpenAISTTProvider) emitUsage(ctx context.Context, res *sttResponse, size int64, filename string) {
	if p.meter == nil {
		return
	}
	u := res.usage()
	audioIn, textIn := u.inputSplit()
	md := map[string]any{}
	var seconds float64
	from := "duration"
	switch {
	case u.Seconds > 0: // what a model billed by duration bills
		seconds = u.Seconds
	case res.Duration > 0:
		seconds = res.Duration
	case audioIn > 0:
		seconds, from = float64(audioIn)/sttAudioTokensPerSecond, "audio_tokens"
	default:
		seconds, from = estimateAudioSeconds(size, filename), "size"
		md["estimated"] = true
	}
	md["seconds"], md["seconds_from"] = seconds, from
	ev := UsageEvent{
		CallerID:    MeterCallerIDFromCtx(ctx),
		Provider:    "openai",
		Model:       p.model,
		Operation:   "stt",
		DebugSpanID: DebugSpanIDFromCtx(ctx),
	}
	priced := false
	if u.billedByToken() {
		ev.InputTokens, ev.OutputTokens, ev.TotalTokens = u.InputTokens, u.OutputTokens, u.TotalTokens
		if ev.TotalTokens == 0 {
			ev.TotalTokens = u.InputTokens + u.OutputTokens
		}
		md["audio_input_tok"], md["text_input_tok"] = audioIn, textIn
		ev.EstimatedCostUSD, priced = estimateSTTCostByTokens(p.model, audioIn, textIn, u.OutputTokens)
	}
	if !priced {
		ev.EstimatedCostUSD = EstimateSTTCost(p.model, seconds)
	}
	ev.Metadata = mergeMeterMetadata(ctx, md)
	p.meter(ev)
}

// estimateAudioSeconds guesses a recording's length from its size when the
// API reports no duration: what browsers record (Opus in WebM or Ogg) runs
// near 32 kbps, compressed music formats near 128 kbps, WAV is 16 kHz mono PCM.
func estimateAudioSeconds(size int64, filename string) float64 {
	rate := 16000.0 // bytes per second, ~128 kbps (mp3, m4a, mp4)
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".webm", ".ogg", ".oga", ".opus":
		rate = 4000
	case ".wav":
		rate = 32000
	case ".flac":
		rate = 24000
	}
	return float64(size) / rate
}
