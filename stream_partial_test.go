package ai

import (
	"context"
	"errors"
	"testing"
)

// meterTap collects the usage events a provider reports.
type meterTap struct{ events []UsageEvent }

func (m *meterTap) hook(ev UsageEvent) { m.events = append(m.events, ev) }

func (m *meterTap) one(t *testing.T) UsageEvent {
	t.Helper()
	if len(m.events) != 1 {
		t.Fatalf("got %d usage events, want exactly 1: %+v", len(m.events), m.events)
	}
	return m.events[0]
}

var errStopped = errors.New("viewer hit stop")

// A user who stops a reply mid-stream still cost the prompt and what was
// written: the event reports both, marked partial, the output estimated.
func TestAnthropicChatStream_StoppedStillBilled(t *testing.T) {
	p := anthropicTestProvider(sseServer(t, anthropicStream("end_turn", true)))
	tap := &meterTap{}
	p.SetMeter(tap.hook)
	_, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "write two files"}}, nil,
		func(StreamChunk) error { return errStopped })
	if !errors.Is(err, errStopped) {
		t.Fatalf("err = %v", err)
	}
	ev := tap.one(t)
	if ev.InputTokens != 10 {
		t.Fatalf("input = %d, want the 10 message_start reported", ev.InputTokens)
	}
	if ev.OutputTokens != CountTokens("Writing both.") || ev.OutputTokens == 0 {
		t.Fatalf("output = %d, want the streamed text counted", ev.OutputTokens)
	}
	if ev.Metadata["partial"] != true || ev.Metadata["estimated"] != true {
		t.Fatalf("metadata = %v, want partial and estimated", ev.Metadata)
	}
}

// A reply that runs to the end reports the provider's own counts, once.
func TestAnthropicChatStream_CompleteBilledOnce(t *testing.T) {
	p := anthropicTestProvider(sseServer(t, anthropicStream("tool_use", true)))
	tap := &meterTap{}
	p.SetMeter(tap.hook)
	if _, err := p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil, func(StreamChunk) error { return nil }); err != nil {
		t.Fatal(err)
	}
	ev := tap.one(t)
	if ev.InputTokens != 10 || ev.OutputTokens != 64 || ev.Metadata["partial"] != nil || ev.Metadata["estimated"] != nil {
		t.Fatalf("event = %+v", ev)
	}
}

// A body that ends before message_delta (a dropped connection) still bills
// what streamed.
func TestAnthropicChatStream_TruncatedBody(t *testing.T) {
	events := anthropicEvent("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":1}}}`) +
		anthropicEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		anthropicEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Half a sentence"}}`)
	p := anthropicTestProvider(sseServer(t, events))
	tap := &meterTap{}
	p.SetMeter(tap.hook)
	_, _ = p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil, func(StreamChunk) error { return nil })
	ev := tap.one(t)
	if ev.InputTokens != 7 || ev.OutputTokens == 0 || ev.Metadata["estimated"] != true {
		t.Fatalf("event = %+v", ev)
	}
}

// OpenAI sends usage only in the last chunk, which a stopped stream never
// gets: the prompt and what streamed are estimated.
func TestOpenAIChatStream_StoppedStillBilled(t *testing.T) {
	events := openAIChunk(`{"role":"assistant","content":"Hello there"}`, "") +
		openAIChunk(`{"content":", and more"}`, "") +
		openAIChunk(`{}`, "stop") +
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}` + "\n\n" +
		"data: [DONE]\n\n"
	prompt := []Message{{Role: RoleSystem, Content: "be brief"}, {Role: RoleUser, Content: "say hello to the whole wide world"}}

	stopped := NewOpenAIProviderWithBaseURL("k", "gpt-test", sseServer(t, events).URL)
	tap := &meterTap{}
	stopped.SetMeter(tap.hook)
	if _, err := stopped.ChatStream(context.Background(), prompt, nil, func(StreamChunk) error { return errStopped }); !errors.Is(err, errStopped) {
		t.Fatalf("err = %v", err)
	}
	ev := tap.one(t)
	if ev.InputTokens != promptTokens(prompt) || ev.OutputTokens != CountTokens("Hello there") {
		t.Fatalf("in/out = %d/%d, want %d/%d", ev.InputTokens, ev.OutputTokens, promptTokens(prompt), CountTokens("Hello there"))
	}
	if ev.Metadata["partial"] != true || ev.Metadata["estimated"] != true {
		t.Fatalf("metadata = %v", ev.Metadata)
	}

	// run to the end: the provider's counts, once, unmarked
	done := NewOpenAIProviderWithBaseURL("k", "gpt-test", sseServer(t, events).URL)
	tap = &meterTap{}
	done.SetMeter(tap.hook)
	if _, err := done.ChatStream(context.Background(), prompt, nil, func(StreamChunk) error { return nil }); err != nil {
		t.Fatal(err)
	}
	ev = tap.one(t)
	if ev.InputTokens != 12 || ev.OutputTokens != 5 || len(ev.Metadata) != 0 {
		t.Fatalf("event = %+v", ev)
	}
}

// A request refused before any output is not billed, so nothing is reported.
func TestOpenAIChatStream_RefusedNotBilled(t *testing.T) {
	p := NewOpenAIProviderWithBaseURL("k", "gpt-test", sseServer(t, "data: [DONE]\n\n").URL)
	tap := &meterTap{}
	p.SetMeter(tap.hook)
	_, _ = p.ChatStream(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil, func(StreamChunk) error { return nil })
	if len(tap.events) != 0 {
		t.Fatalf("an empty stream reported usage: %+v", tap.events)
	}
}
