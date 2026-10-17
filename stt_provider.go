package ai

import (
	"context"
	"io"
	"strings"
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

type ctxKeySTTLanguage struct{}

// WithSTTLanguage hints the spoken language of the audio to Transcribe, as a
// BCP-47 tag ("en-US") or a bare language ("en"). Each provider gets the
// form it takes: the primary subtag, "en". Without it, or with a tag that
// does not parse, the provider detects the language itself, which can pick a
// wrong one for a short or unclear clip. A provider with no such parameter
// ignores the hint.
func WithSTTLanguage(ctx context.Context, tag string) context.Context {
	return context.WithValue(ctx, ctxKeySTTLanguage{}, tag)
}

// STTLanguageFromCtx is the primary language subtag of the hint set with
// WithSTTLanguage, lowercased ("en-US" gives "en"); "" when there is none or
// it is not a 2 or 3 letter code.
func STTLanguageFromCtx(ctx context.Context) string {
	tag, _ := ctx.Value(ctxKeySTTLanguage{}).(string)
	lang, _, _ := strings.Cut(strings.NewReplacer("_", "-").Replace(strings.TrimSpace(tag)), "-")
	if len(lang) < 2 || len(lang) > 3 {
		return ""
	}
	lang = strings.ToLower(lang)
	for _, r := range lang {
		if r < 'a' || r > 'z' {
			return ""
		}
	}
	return lang
}
