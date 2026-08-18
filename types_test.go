package ai

import (
	"encoding/json"
	"testing"
)

func TestPart_TypeSwitch(t *testing.T) {
	parts := []Part{
		TextPart{Text: "hello"},
		ImagePart{MediaType: "image/png", Data: "AAAA"},
	}
	if parts[0].partKind() != "text" {
		t.Errorf("TextPart.partKind() = %q", parts[0].partKind())
	}
	if parts[1].partKind() != "image" {
		t.Errorf("ImagePart.partKind() = %q", parts[1].partKind())
	}
	switch v := parts[0].(type) {
	case TextPart:
		if v.Text != "hello" {
			t.Errorf("TextPart.Text = %q", v.Text)
		}
	default:
		t.Errorf("expected TextPart, got %T", parts[0])
	}
	switch v := parts[1].(type) {
	case ImagePart:
		if v.MediaType != "image/png" || v.Data != "AAAA" {
			t.Errorf("ImagePart = %+v", v)
		}
	default:
		t.Errorf("expected ImagePart, got %T", parts[1])
	}
}

func TestMessage_JSONRoundTrip_TextOnly(t *testing.T) {
	orig := Message{
		Role:    RoleUser,
		Content: "hi",
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	var got Message
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Role != orig.Role || got.Content != orig.Content {
		t.Errorf("round-trip mismatch: %+v", got)
	}
	if len(got.Parts) != 0 {
		t.Errorf("expected no parts, got %d", len(got.Parts))
	}
}

func TestMessage_JSONRoundTrip_WithParts(t *testing.T) {
	orig := Message{
		Role: RoleUser,
		Parts: []Part{
			TextPart{Text: "describe"},
			ImagePart{MediaType: "image/jpeg", Data: "QUJD"},
		},
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	var got Message
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(got.Parts))
	}
	tp, ok := got.Parts[0].(TextPart)
	if !ok || tp.Text != "describe" {
		t.Errorf("part[0] = %+v, want TextPart{describe}", got.Parts[0])
	}
	ip, ok := got.Parts[1].(ImagePart)
	if !ok || ip.MediaType != "image/jpeg" || ip.Data != "QUJD" {
		t.Errorf("part[1] = %+v, want ImagePart{jpeg, QUJD}", got.Parts[1])
	}
}

func TestMessage_JSONRoundTrip_WithToolCalls(t *testing.T) {
	orig := Message{
		Role:    RoleAssistant,
		Content: "sure",
		ToolCalls: []ToolCall{
			{ID: "tc1", Name: "search", Arguments: json.RawMessage(`{"q":"x"}`)},
		},
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatal(err)
	}
	var got Message
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.ToolCalls) != 1 || got.ToolCalls[0].Name != "search" {
		t.Errorf("tool calls mismatch: %+v", got.ToolCalls)
	}
}
