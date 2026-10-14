package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// DefaultElevenLabsBaseURL is the ElevenLabs API root.
const DefaultElevenLabsBaseURL = "https://api.elevenlabs.io"

// elevenLabsClient is the HTTP plumbing shared by the ElevenLabs TTS, STT
// and sound-effect providers: xi-api-key auth, error decoding.
type elevenLabsClient struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

func newElevenLabsClient(apiKey string) *elevenLabsClient {
	return &elevenLabsClient{apiKey: apiKey, baseURL: DefaultElevenLabsBaseURL, http: http.DefaultClient}
}

// post sends body to path and returns the response on 2xx; any other status
// is decoded into an error and the body closed.
func (c *elevenLabsClient) post(ctx context.Context, path string, body io.Reader, contentType string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(c.baseURL, "/")+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("xi-api-key", c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		return resp, nil
	}
	defer resp.Body.Close() //nolint:errcheck
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return nil, elevenLabsError(resp.StatusCode, raw)
}

func (c *elevenLabsClient) postJSON(ctx context.Context, path string, payload any) (*http.Response, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return c.post(ctx, path, bytes.NewReader(b), "application/json")
}

// elevenLabsError reads {"detail":{"code","message"}}, or a plain detail
// string, falling back to the raw body.
func elevenLabsError(status int, raw []byte) error {
	var body struct {
		Detail json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(raw, &body) == nil && len(body.Detail) > 0 {
		var d struct {
			Code    string `json:"code"`
			Status  string `json:"status"`
			Message string `json:"message"`
		}
		if json.Unmarshal(body.Detail, &d) == nil && d.Message != "" {
			code := d.Code
			if code == "" {
				code = d.Status
			}
			return fmt.Errorf("http %d %s: %s", status, code, d.Message)
		}
		var s string
		if json.Unmarshal(body.Detail, &s) == nil && s != "" {
			return fmt.Errorf("http %d: %s", status, s)
		}
	}
	return fmt.Errorf("http %d: %s", status, strings.TrimSpace(string(raw)))
}

// elevenLabsOutputFormat maps a container name (mp3, pcm, opus, wav) to an
// ElevenLabs output_format; a full ElevenLabs format ("mp3_22050_32",
// "ulaw_8000", …) passes through. aac and flac are not offered.
func elevenLabsOutputFormat(format string) (string, error) {
	switch format {
	case "", "mp3":
		return "mp3_44100_128", nil
	case "pcm":
		return "pcm_24000", nil
	case "opus":
		return "opus_48000_128", nil
	case "wav":
		return "wav_44100", nil
	}
	if strings.Contains(format, "_") {
		return format, nil
	}
	return "", fmt.Errorf("unsupported format %q (mp3, pcm, opus, wav, or an ElevenLabs output_format)", format)
}

// elevenLabsAudioSeconds estimates the length of size bytes of audio in an
// ElevenLabs output_format: mp3/opus from the bitrate, pcm from the sample
// rate (16-bit mono). 0 when the format does not say.
func elevenLabsAudioSeconds(outputFormat string, size int64) float64 {
	parts := strings.Split(outputFormat, "_")
	switch {
	case len(parts) == 3 && (parts[0] == "mp3" || parts[0] == "opus"):
		if kbps, err := strconv.Atoi(parts[2]); err == nil && kbps > 0 {
			return float64(size) * 8 / float64(kbps*1000)
		}
	case len(parts) == 2 && parts[0] == "pcm":
		if rate, err := strconv.Atoi(parts[1]); err == nil && rate > 0 {
			return float64(size) / float64(rate*2)
		}
	}
	return 0
}
