package ai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

type streamingStub struct {
	stubProvider
	chunks []string
	err    error
}

func (s *streamingStub) ChatStream(_ context.Context, _ []Message, _ []Tool, cb func(StreamChunk) error) (*Response, error) {
	if s.err != nil {
		return nil, s.err
	}
	var content strings.Builder
	for _, c := range s.chunks {
		content.WriteString(c)
		if err := cb(StreamChunk{Text: c}); err != nil {
			return &Response{Content: content.String()}, err
		}
	}
	return &Response{Content: content.String()}, nil
}

func TestStream_Iter(t *testing.T) {
	p := &streamingStub{chunks: []string{"hel", "lo ", "world"}}
	chunks, result := Stream(context.Background(), p, []Message{{Role: RoleUser, Content: "hi"}}, nil)

	var got strings.Builder
	for chunk, err := range chunks {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	if got.String() != "hello world" {
		t.Errorf("got %q, want %q", got.String(), "hello world")
	}

	resp, err := result.Response()
	if err != nil {
		t.Fatalf("response error: %v", err)
	}
	if resp.Content != "hello world" {
		t.Errorf("response content %q, want %q", resp.Content, "hello world")
	}
}

func TestStream_Break(t *testing.T) {
	p := &streamingStub{chunks: []string{"a", "b", "c", "d", "e"}}
	chunks, result := Stream(context.Background(), p, nil, nil)

	var count int
	for _, err := range chunks {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++
		if count == 2 {
			break
		}
	}
	if count != 2 {
		t.Errorf("got %d chunks, want 2", count)
	}

	resp, err := result.Response()
	if err != nil {
		t.Fatalf("response error: %v", err)
	}
	if resp.Content != "ab" {
		t.Errorf("response content %q, want %q", resp.Content, "ab")
	}
}

func TestStream_ProviderError(t *testing.T) {
	p := &streamingStub{err: fmt.Errorf("provider down")}
	chunks, result := Stream(context.Background(), p, nil, nil)

	var sawErr bool
	for _, err := range chunks {
		if err != nil {
			sawErr = true
			if !strings.Contains(err.Error(), "provider down") {
				t.Errorf("unexpected error: %v", err)
			}
		}
	}
	if !sawErr {
		t.Fatal("expected error chunk")
	}

	_, err := result.Response()
	if err == nil || !strings.Contains(err.Error(), "provider down") {
		t.Errorf("expected provider down error, got: %v", err)
	}
}

func TestStream_FallbackToChat(t *testing.T) {
	p := &stubProvider{chatResponse: &Response{Content: "fallback response"}}
	chunks, result := Stream(context.Background(), p, nil, nil)

	var got strings.Builder
	for chunk, err := range chunks {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	if got.String() != "fallback response" {
		t.Errorf("got %q, want %q", got.String(), "fallback response")
	}

	resp, err := result.Response()
	if err != nil {
		t.Fatalf("response error: %v", err)
	}
	if resp.Content != "fallback response" {
		t.Errorf("response content %q", resp.Content)
	}
}

func TestStream_FallbackError(t *testing.T) {
	p := &stubProvider{err: fmt.Errorf("chat fail")}
	chunks, result := Stream(context.Background(), p, nil, nil)

	for _, err := range chunks {
		if err == nil {
			t.Fatal("expected error from fallback")
		}
	}
	_, err := result.Response()
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestStreamWithChan_Basic(t *testing.T) {
	p := &streamingStub{chunks: []string{"one", "two"}}
	ch, wait := StreamWithChan(context.Background(), p, nil, nil)

	var got strings.Builder
	for chunk := range ch {
		got.WriteString(chunk.Text)
	}
	if got.String() != "onetwo" {
		t.Errorf("got %q", got.String())
	}

	resp, err := wait()
	if err != nil {
		t.Fatalf("wait error: %v", err)
	}
	if resp.Content != "onetwo" {
		t.Errorf("response content %q", resp.Content)
	}
}

func TestStreamWithChan_Fallback(t *testing.T) {
	p := &stubProvider{chatResponse: &Response{Content: "chan fallback"}}
	ch, wait := StreamWithChan(context.Background(), p, nil, nil)

	var got strings.Builder
	for chunk := range ch {
		got.WriteString(chunk.Text)
	}
	if got.String() != "chan fallback" {
		t.Errorf("got %q", got.String())
	}

	resp, err := wait()
	if err != nil {
		t.Fatalf("wait error: %v", err)
	}
	if resp.Content != "chan fallback" {
		t.Errorf("response content %q", resp.Content)
	}
}

func TestStreamWithChan_Error(t *testing.T) {
	p := &streamingStub{err: errors.New("boom")}
	ch, wait := StreamWithChan(context.Background(), p, nil, nil)

	for range ch {
		t.Fatal("expected no chunks")
	}

	_, err := wait()
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("expected boom, got: %v", err)
	}
}

func TestStreamWithChan_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &streamingStub{chunks: []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}}
	ch, wait := StreamWithChan(ctx, p, nil, nil)

	<-ch
	cancel()

	for range ch {
	}

	_, err := wait()
	if err != nil && !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}
}

func TestExtractPrompts(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "sys1"},
		{Role: RoleSystem, Content: "sys2"},
		{Role: RoleUser, Content: "user1"},
		{Role: RoleAssistant, Content: "ignore"},
		{Role: RoleUser, Content: "user2"},
	}
	sys, user := extractPrompts(msgs)
	if sys != "sys1\n\nsys2" {
		t.Errorf("sys = %q", sys)
	}
	if user != "user1\n\nuser2" {
		t.Errorf("user = %q", user)
	}
}

func TestStreamResult_MultipleResponse(t *testing.T) {
	r := &StreamResult{done: make(chan struct{})}
	r.set(&Response{Content: "first"}, nil)
	r.set(&Response{Content: "second"}, fmt.Errorf("late"))

	resp, err := r.Response()
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "first" {
		t.Errorf("expected first, got %q", resp.Content)
	}
}
