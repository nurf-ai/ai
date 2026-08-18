// Package ai is the consumer-facing LLM contract — provider-agnostic types
// (Tool/Message/Response/ToolCall/Part) plus the LLMProvider interface
// in ai_contract.go. Concrete implementations (Anthropic/OpenAI/Ollama/
// HuggingFace) live in sibling files; they satisfy LLMProvider structurally.
package ai

import (
	"encoding/json"
	"strings"
)

// Role constants used in Message.Role.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Tool defines a provider-agnostic tool the model can call.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema object
}

// ToolCall represents the model requesting a tool invocation.
type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// Part is a single element of a multimodal message. Sealed — only TextPart
// and ImagePart satisfy this interface.
type Part interface {
	partKind() string
}

// TextPart carries inline text.
type TextPart struct {
	Text string
}

func (TextPart) partKind() string { return "text" }

// ImagePart carries base64-encoded image data.
type ImagePart struct {
	MediaType string // e.g. "image/png"
	Data      string // base64-encoded
}

func (ImagePart) partKind() string { return "image" }

// partWire is the JSON shape for []Part on the wire (backwards-compat with
// the old ContentPart struct).
type partWire struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// CacheControl tells a provider to mark this message (or its last
// content block) as a cache breakpoint. Only Anthropic acts on it
// today — other providers ignore it silently.
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// Message is a provider-agnostic chat message.
type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	Parts        []Part        `json:"-"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// messageWire mirrors Message for JSON but uses partWire for Parts.
type messageWire struct {
	Role         string        `json:"role"`
	Content      string        `json:"content,omitempty"`
	Parts        []partWire    `json:"parts,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (m Message) MarshalJSON() ([]byte, error) {
	w := messageWire{
		Role:         m.Role,
		Content:      m.Content,
		ToolCalls:    m.ToolCalls,
		ToolCallID:   m.ToolCallID,
		CacheControl: m.CacheControl,
	}
	for _, p := range m.Parts {
		switch v := p.(type) {
		case TextPart:
			w.Parts = append(w.Parts, partWire{Type: "text", Text: v.Text})
		case ImagePart:
			w.Parts = append(w.Parts, partWire{Type: "image", MediaType: v.MediaType, Data: v.Data})
		}
	}
	return json.Marshal(w)
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var w messageWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	m.Role = w.Role
	m.Content = w.Content
	m.ToolCalls = w.ToolCalls
	m.ToolCallID = w.ToolCallID
	m.CacheControl = w.CacheControl
	m.Parts = nil
	for _, pw := range w.Parts {
		switch pw.Type {
		case "image":
			m.Parts = append(m.Parts, ImagePart{MediaType: pw.MediaType, Data: pw.Data})
		default:
			m.Parts = append(m.Parts, TextPart{Text: pw.Text})
		}
	}
	return nil
}

// Response is the provider-agnostic result of a Chat call.
type Response struct {
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// extractJSONObject tries to pull a JSON object from text that may be
// wrapped in markdown fences or preceded by prose.
func extractJSONObject(s string) ([]byte, bool) {
	s = strings.TrimSpace(s)
	if json.Valid([]byte(s)) && strings.HasPrefix(s, "{") {
		return []byte(s), true
	}
	// Strip markdown fences.
	if strings.HasPrefix(s, "```") {
		lines := strings.SplitN(s, "\n", 2)
		if len(lines) == 2 {
			s = lines[1]
		}
		if idx := strings.LastIndex(s, "```"); idx >= 0 {
			s = s[:idx]
		}
		s = strings.TrimSpace(s)
		if json.Valid([]byte(s)) && strings.HasPrefix(s, "{") {
			return []byte(s), true
		}
	}
	// Last resort: first '{' to last '}'.
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		candidate := s[start : end+1]
		if json.Valid([]byte(candidate)) {
			return []byte(candidate), true
		}
	}
	return nil, false
}

// stripThinkTags removes <think>…</think> blocks emitted by thinking models.
func stripThinkTags(s string) string {
	for {
		start := strings.Index(s, "<think>")
		if start < 0 {
			break
		}
		end := strings.Index(s, "</think>")
		if end < 0 {
			s = s[:start]
			break
		}
		s = s[:start] + s[end+len("</think>"):]
	}
	return strings.TrimSpace(s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
