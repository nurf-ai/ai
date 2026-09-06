package ai

import (
	"context"
	"testing"
)

func TestNewLLMProvider_AllNames(t *testing.T) {
	for _, name := range []string{"anthropic", "openai", "ollama", "huggingface", "unknown-defaults-to-anthropic"} {
		p := NewLLMProvider(name, "fake-key", "fake-model")
		if p == nil {
			t.Errorf("NewLLMProvider(%q) returned nil", name)
		}
	}
}

func TestNewLLMProvider_WithOptions(t *testing.T) {
	var hooked bool
	hook := func(_ UsageEvent) { hooked = true }
	p := NewLLMProvider("openai", "fake-key", "gpt-4o", WithMeterOption(hook))
	if p == nil {
		t.Fatal("nil provider")
	}
	if m, ok := p.(interface{ SetMeter(MeterHook) }); ok {
		_ = m
	}
	_ = hooked
}

func TestNewImageProvider_OpenAI(t *testing.T) {
	p, err := NewImageProvider(context.Background(), "openai", "fake-key", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p == nil {
		t.Fatal("nil provider")
	}
}

func TestNewImageProvider_UnsupportedProvider(t *testing.T) {
	_, err := NewImageProvider(context.Background(), "nope", "k", "m")
	if err == nil {
		t.Fatal("expected error for unsupported provider")
	}
}

func TestNewSTTProvider_OpenAI(t *testing.T) {
	p := NewSTTProvider("openai", "fake-key", "")
	if p == nil {
		t.Fatal("nil provider")
	}
}

func TestNewSTTProvider_Unsupported(t *testing.T) {
	p := NewSTTProvider("nope", "k", "m")
	if p != nil {
		t.Errorf("expected nil for unsupported provider, got %v", p)
	}
}

func TestNewTTSProvider_OpenAI(t *testing.T) {
	p := NewTTSProvider("openai", "fake-key", "")
	if p == nil {
		t.Fatal("expected non-nil TTS provider")
	}
	if _, ok := p.(*OpenAITTSProvider); !ok {
		t.Errorf("expected *OpenAITTSProvider, got %T", p)
	}
	if got := p.(*OpenAITTSProvider).model; got != "gpt-4o-mini-tts" {
		t.Errorf("default model = %q, want gpt-4o-mini-tts", got)
	}
}

func TestNewTTSProvider_Unsupported(t *testing.T) {
	p := NewTTSProvider("nope", "k", "m")
	if p != nil {
		t.Errorf("expected nil for unsupported provider, got %v", p)
	}
}

func TestTTSMimeType(t *testing.T) {
	cases := map[string]string{"": "audio/mpeg", "mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg", "bogus": "application/octet-stream"}
	for format, want := range cases {
		if got := TTSMimeType(format); got != want {
			t.Errorf("TTSMimeType(%q) = %q, want %q", format, got, want)
		}
	}
}

func TestNewEmbedder_OpenAI(t *testing.T) {
	e := NewEmbedder("openai", "fake-key")
	if e == nil {
		t.Fatal("nil embedder")
	}
}

func TestNewEmbedder_Unsupported(t *testing.T) {
	e := NewEmbedder("nope", "k")
	if e != nil {
		t.Errorf("expected nil for unsupported provider, got %v", e)
	}
}
