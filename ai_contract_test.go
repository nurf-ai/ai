package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

// stubProvider is a minimal LLMProvider for unit tests — returns canned
// structured output and chat responses without hitting any external API.
type stubProvider struct {
	name  string
	model string
	// structuredResponse is marshaled into the output struct on CreateStructuredOutput.
	structuredResponse any
	chatResponse       *Response
	err                error
}

func (s *stubProvider) Name() string  { return s.name }
func (s *stubProvider) Model() string { return s.model }

func (s *stubProvider) CreateStructuredOutput(_ context.Context, _, _ string, out any) error {
	if s.err != nil {
		return s.err
	}
	b, err := json.Marshal(s.structuredResponse)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, out)
}

func (s *stubProvider) CreateStructuredOutputFromSchema(_ context.Context, _, _ string, _ json.RawMessage) (map[string]any, error) {
	return nil, fmt.Errorf("not implemented in stub")
}

func (s *stubProvider) Chat(_ context.Context, _ []Message, _ []Tool) (*Response, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.chatResponse, nil
}

// --- GenerateTyped tests ---

type sampleOutput struct {
	Name  string `json:"name"`
	Score int    `json:"score"`
}

func TestGenerateTyped_Success(t *testing.T) {
	prov := &stubProvider{
		name:               "stub",
		model:              "test-model",
		structuredResponse: sampleOutput{Name: "alice", Score: 42},
	}
	got, err := GenerateTyped[sampleOutput](context.Background(), prov, "prompt", "sys")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name != "alice" || got.Score != 42 {
		t.Errorf("got %+v, want {alice 42}", got)
	}
}

func TestGenerateTyped_Error(t *testing.T) {
	prov := &stubProvider{err: fmt.Errorf("boom")}
	_, err := GenerateTyped[sampleOutput](context.Background(), prov, "p", "s")
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "boom" {
		t.Errorf("got %q, want %q", err.Error(), "boom")
	}
}

func TestGenerateTyped_MapOutput(t *testing.T) {
	prov := &stubProvider{
		structuredResponse: map[string]any{"key": "val"},
	}
	got, err := GenerateTyped[map[string]any](context.Background(), prov, "p", "s")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if (*got)["key"] != "val" {
		t.Errorf("got %v, want map with key=val", got)
	}
}

// --- Middleware / Chain tests ---

func TestChain_OrderOuterFirst(t *testing.T) {
	var order []string
	mw := func(tag string) Middleware {
		return func(next ModelFunc) ModelFunc {
			return func(ctx context.Context, msgs []Message, tools []Tool) (*Response, error) {
				order = append(order, tag+":before")
				resp, err := next(ctx, msgs, tools)
				order = append(order, tag+":after")
				return resp, err
			}
		}
	}
	inner := func(_ context.Context, _ []Message, _ []Tool) (*Response, error) {
		order = append(order, "inner")
		return &Response{Content: "ok"}, nil
	}
	chained := Chain(mw("a"), mw("b"))(inner)
	resp, err := chained(context.Background(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "ok" {
		t.Errorf("content = %q", resp.Content)
	}
	want := []string{"a:before", "b:before", "inner", "b:after", "a:after"}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Errorf("order[%d] = %q, want %q", i, order[i], want[i])
		}
	}
}

func TestChain_Empty(t *testing.T) {
	inner := func(_ context.Context, _ []Message, _ []Tool) (*Response, error) {
		return &Response{Content: "direct"}, nil
	}
	chained := Chain()(inner)
	resp, _ := chained(context.Background(), nil, nil)
	if resp.Content != "direct" {
		t.Errorf("expected passthrough, got %q", resp.Content)
	}
}

// --- SetLLMMeter interface-based dispatch tests ---

type meterableProvider struct {
	stubProvider
	hook MeterHook
}

func (p *meterableProvider) SetMeter(hook MeterHook) { p.hook = hook }

func TestSetLLMMeter_InterfaceBased(t *testing.T) {
	p := &meterableProvider{}
	var called bool
	hook := func(_ UsageEvent) { called = true }
	SetLLMMeter(p, hook)
	if p.hook == nil {
		t.Fatal("SetMeter not called")
	}
	p.hook(UsageEvent{})
	if !called {
		t.Error("hook not invoked")
	}
}

func TestSetLLMMeter_NilHookNoop(t *testing.T) {
	p := &meterableProvider{}
	SetLLMMeter(p, nil)
	if p.hook != nil {
		t.Error("nil hook should be a no-op")
	}
}

func TestSetLLMMeter_NonMeterableNoop(t *testing.T) {
	p := &stubProvider{}
	SetLLMMeter(p, func(_ UsageEvent) {})
}

// --- SetLLMModeration interface-based dispatch tests ---

type moderableProvider struct {
	stubProvider
	mod ModerationProvider
}

func (p *moderableProvider) SetModeration(m ModerationProvider) { p.mod = m }

func TestSetLLMModeration_InterfaceBased(t *testing.T) {
	p := &moderableProvider{}
	SetLLMModeration(p, NewModerationProvider("openai", "fake"))
	if p.mod == nil {
		t.Fatal("SetModeration not called")
	}
}

func TestSetLLMModeration_NilNoop(t *testing.T) {
	p := &moderableProvider{}
	SetLLMModeration(p, nil)
	if p.mod != nil {
		t.Error("nil moderation should be no-op")
	}
}
