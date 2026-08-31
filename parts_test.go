package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

// mmStub implements MultimodalStructuredProvider on top of stubProvider.
type mmStub struct {
	stubProvider
	gotParts []Part
	gotSys   string
}

func (m *mmStub) CreateStructuredOutputFromParts(_ context.Context, parts []Part, sysPrompt string, _ json.RawMessage) (map[string]any, error) {
	m.gotParts = parts
	m.gotSys = sysPrompt
	return map[string]any{"ok": true}, nil
}

// textOnlyStub records the flattened prompt it received.
type textOnlyStub struct {
	stubProvider
	gotUser string
}

func (s *textOnlyStub) CreateStructuredOutputFromSchema(_ context.Context, user, _ string, _ json.RawMessage) (map[string]any, error) {
	s.gotUser = user
	return map[string]any{"text": user}, nil
}

func TestPartsText(t *testing.T) {
	text, hasImage := PartsText([]Part{
		TextPart{Text: "one"},
		ImagePart{MediaType: "image/png", Data: "AAAA"},
		TextPart{Text: "two"},
	})
	if text != "one\n\ntwo" {
		t.Errorf("text = %q", text)
	}
	if !hasImage {
		t.Error("hasImage = false, want true")
	}
	if _, has := PartsText(nil); has {
		t.Error("empty parts reported an image")
	}
}

func TestStructuredOutputFromParts_RoutesToMultimodal(t *testing.T) {
	p := &mmStub{}
	parts := []Part{ImagePart{MediaType: "image/jpeg", Data: "QUJD"}, TextPart{Text: "describe"}}
	out, err := StructuredOutputFromParts(context.Background(), p, parts, "sys", json.RawMessage(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out["ok"] != true {
		t.Errorf("out = %v", out)
	}
	if len(p.gotParts) != 2 || p.gotSys != "sys" {
		t.Errorf("parts/sys not forwarded verbatim: %d parts, sys=%q", len(p.gotParts), p.gotSys)
	}
}

func TestStructuredOutputFromParts_TextFallback(t *testing.T) {
	p := &textOnlyStub{}
	out, err := StructuredOutputFromParts(context.Background(), p, []Part{TextPart{Text: "a"}, TextPart{Text: "b"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.gotUser != "a\n\nb" || out["text"] != "a\n\nb" {
		t.Errorf("fallback did not flatten text parts: got %q", p.gotUser)
	}
}

func TestStructuredOutputFromParts_RefusesToDropImages(t *testing.T) {
	p := &textOnlyStub{}
	_, err := StructuredOutputFromParts(context.Background(), p, []Part{ImagePart{MediaType: "image/png", Data: "AA=="}}, "", nil)
	if !errors.Is(err, ErrVisionUnsupported) {
		t.Fatalf("err = %v, want ErrVisionUnsupported", err)
	}
	if p.gotUser != "" {
		t.Error("text fallback ran despite an image part")
	}
}
