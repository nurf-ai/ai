package ai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// ErrFileUnsupported is returned when a FilePart reaches a provider (or a
// media type) that cannot take it natively. The part is never silently
// dropped. Check AcceptsFile first to avoid it.
var ErrFileUnsupported = errors.New("ai: provider does not accept this file input")

// ErrFileTooLarge is returned when the files of a request exceed what the
// provider takes inline.
var ErrFileTooLarge = errors.New("ai: file input exceeds the provider's inline size limit")

// FileAcceptor is implemented by providers that can say which FilePart media
// types they take natively. Every built-in chat provider implements it.
type FileAcceptor interface {
	AcceptsFile(mediaType string) bool
}

// AcceptsFile reports whether p takes a FilePart of mediaType natively. A
// provider that does not implement FileAcceptor (a third-party or wrapped
// one) reports false. Parameters in mediaType ("; codecs=opus") are ignored.
func AcceptsFile(p LLMProvider, mediaType string) bool {
	fa, ok := p.(FileAcceptor)
	return ok && fa.AcceptsFile(mediaType)
}

// PartsHaveFile reports whether any part is a FilePart.
func PartsHaveFile(parts []Part) bool {
	for _, p := range parts {
		if _, ok := p.(FilePart); ok {
			return true
		}
	}
	return false
}

// normalizeMediaType lowercases mt and drops parameters.
func normalizeMediaType(mt string) string {
	mt, _, _ = strings.Cut(mt, ";")
	return strings.ToLower(strings.TrimSpace(mt))
}

// geminiInlineLimit is Gemini's inline request cap (~20 MB for the whole
// request); the files of one request are held under it.
const geminiInlineLimit = 20 << 20

// checkFileParts fails with ErrFileUnsupported on the first FilePart whose
// media type accepts rejects, and with ErrFileTooLarge when maxBytes > 0 and
// the files together exceed it.
func checkFileParts(msgs []Message, accepts func(string) bool, maxBytes int) error {
	total := 0
	for _, m := range msgs {
		for _, p := range m.Parts {
			if f, ok := p.(FilePart); ok {
				if !accepts(f.MediaType) {
					return fmt.Errorf("%w: %q", ErrFileUnsupported, f.MediaType)
				}
				total += len(f.Data)
			}
		}
	}
	if maxBytes > 0 && total > maxBytes {
		return fmt.Errorf("%w: %d bytes, limit %d", ErrFileTooLarge, total, maxBytes)
	}
	return nil
}

// checkPartsFiles is checkFileParts for a single turn of parts.
func checkPartsFiles(parts []Part, accepts func(string) bool, maxBytes int) error {
	return checkFileParts([]Message{{Parts: parts}}, accepts, maxBytes)
}

func acceptsNoFile(string) bool { return false }

func anthropicAcceptsFile(mt string) bool { return normalizeMediaType(mt) == "application/pdf" }

func geminiAcceptsFile(mt string) bool {
	mt = normalizeMediaType(mt)
	return mt == "application/pdf" || mt == "video/mp4" || (strings.HasPrefix(mt, "audio/") && len(mt) > len("audio/"))
}

// openaiAudioFormat maps an audio media type to input_audio's format, "" when
// OpenAI does not take it.
func openaiAudioFormat(mt string) string {
	switch normalizeMediaType(mt) {
	case "audio/wav", "audio/x-wav", "audio/wave":
		return "wav"
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	}
	return ""
}

func openaiAcceptsFile(mt string) bool {
	return normalizeMediaType(mt) == "application/pdf" || openaiAudioFormat(mt) != ""
}

// Chat Completions content parts the pinned OpenAI client has no type for
// ("file", "input_audio"). They ride through the client as a placeholder
// text part holding the final JSON; filePartTransport swaps it in on the
// wire.
const openaiFilePlaceholder = "ai_file_part"

// openaiFileChatPart builds the placeholder for a FilePart already checked
// with openaiAcceptsFile.
func openaiFileChatPart(v FilePart) openai.ChatMessagePart {
	b64 := base64.StdEncoding.EncodeToString(v.Data)
	var wire map[string]any
	if f := openaiAudioFormat(v.MediaType); f != "" {
		wire = map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": b64, "format": f}}
	} else {
		name := v.Name
		if name == "" {
			name = "file.pdf"
		}
		wire = map[string]any{"type": "file", "file": map[string]any{
			"filename":  name,
			"file_data": "data:" + normalizeMediaType(v.MediaType) + ";base64," + b64,
		}}
	}
	raw, _ := json.Marshal(wire)
	return openai.ChatMessagePart{Type: openaiFilePlaceholder, Text: string(raw)}
}

// filePartTransport rewrites the placeholder parts of a chat request body
// into the real file / input_audio content parts. Requests without a
// placeholder pass through untouched.
type filePartTransport struct {
	base http.RoundTripper
}

func (t filePartTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	if req.Body == nil || req.Body == http.NoBody {
		return base.RoundTrip(req)
	}
	body, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		return nil, err
	}
	if bytes.Contains(body, []byte(openaiFilePlaceholder)) {
		body = rewriteFileParts(body)
	}
	r := req.Clone(req.Context())
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))
	r.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	return base.RoundTrip(r)
}

func rewriteFileParts(body []byte) []byte {
	var req map[string]json.RawMessage
	if json.Unmarshal(body, &req) != nil {
		return body
	}
	var msgs []map[string]json.RawMessage
	if json.Unmarshal(req["messages"], &msgs) != nil {
		return body
	}
	for _, m := range msgs {
		var parts []map[string]json.RawMessage
		if json.Unmarshal(m["content"], &parts) != nil {
			continue
		}
		changed := false
		for i, p := range parts {
			var typ, text string
			_ = json.Unmarshal(p["type"], &typ)
			if typ != openaiFilePlaceholder {
				continue
			}
			_ = json.Unmarshal(p["text"], &text)
			var real map[string]json.RawMessage
			if json.Unmarshal([]byte(text), &real) == nil {
				parts[i] = real
				changed = true
			}
		}
		if changed {
			m["content"], _ = json.Marshal(parts)
		}
	}
	req["messages"], _ = json.Marshal(msgs)
	out, err := json.Marshal(req)
	if err != nil {
		return body
	}
	return out
}

// withFilePartTransport installs filePartTransport on cfg's HTTP client
// (a copy, so a shared client is left alone).
func withFilePartTransport(cfg openai.ClientConfig) openai.ClientConfig {
	hc, ok := cfg.HTTPClient.(*http.Client)
	if !ok || hc == nil {
		return cfg
	}
	c := *hc
	c.Transport = filePartTransport{base: c.Transport}
	cfg.HTTPClient = &c
	return cfg
}
