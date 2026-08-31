package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
)

// MultimodalStructuredProvider is implemented by providers that accept a
// multimodal user turn (text + images) for schema-constrained structured
// output. Every built-in provider implements it; third-party LLMProvider
// implementations may not, which is why StructuredOutputFromParts exists.
type MultimodalStructuredProvider interface {
	CreateStructuredOutputFromParts(ctx context.Context, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error)
}

// ErrVisionUnsupported is returned when an ImagePart is handed to a provider
// that only accepts text for structured output. The image is never silently
// dropped — a spec authored without the reference it was asked to describe
// is worse than an error.
var ErrVisionUnsupported = errors.New("ai: provider does not accept image input for structured output")

// StructuredOutputFromParts routes a multimodal structured-output request to
// p. Providers implementing MultimodalStructuredProvider receive the parts
// verbatim; any other provider receives the text parts joined, and
// ErrVisionUnsupported when an ImagePart would otherwise be lost.
func StructuredOutputFromParts(ctx context.Context, p LLMProvider, parts []Part, sysPrompt string, schema json.RawMessage) (map[string]any, error) {
	if mp, ok := p.(MultimodalStructuredProvider); ok {
		return mp.CreateStructuredOutputFromParts(ctx, parts, sysPrompt, schema)
	}
	text, hasImage := PartsText(parts)
	if hasImage {
		return nil, ErrVisionUnsupported
	}
	return p.CreateStructuredOutputFromSchema(ctx, text, sysPrompt, schema)
}

// PartsText joins the TextParts of a multimodal turn with blank lines and
// reports whether any ImagePart was present. Used for moderation, usage
// attribution and text-only fallbacks.
func PartsText(parts []Part) (string, bool) {
	var b strings.Builder
	hasImage := false
	for _, p := range parts {
		switch v := p.(type) {
		case TextPart:
			if b.Len() > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(v.Text)
		case ImagePart:
			hasImage = true
		}
	}
	return b.String(), hasImage
}

// imagePartBytes returns the raw image bytes of an ImagePart. Data is base64
// by contract; a payload that fails to decode is passed through untouched so
// a caller that already handed over raw bytes keeps working.
func imagePartBytes(v ImagePart) []byte {
	if b, err := base64.StdEncoding.DecodeString(v.Data); err == nil {
		return b
	}
	return []byte(v.Data)
}
