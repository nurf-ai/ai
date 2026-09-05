package ai

import (
	"context"
	"errors"
	"iter"
	"strings"
	"sync"
)

// StreamChunk is a single piece of a streaming response.
// Text carries free-form content deltas. ToolName + ToolArg carry
// tool-call argument fragments so callers can stream tool output
// (e.g. the REPLY tool's text field) before the call finalizes.
type StreamChunk struct {
	Text     string
	ToolName string
	ToolArg  string
}

var errStreamBreak = errors.New("stream break")

// StreamingProvider extends LLMProvider with streaming chat support.
type StreamingProvider interface {
	ChatStream(ctx context.Context, messages []Message, tools []Tool, cb func(StreamChunk) error) (*Response, error)
}

// StreamResult holds the final accumulated response from a stream.
type StreamResult struct {
	once sync.Once
	done chan struct{}
	resp *Response
	err  error
}

func (r *StreamResult) set(resp *Response, err error) {
	r.once.Do(func() {
		r.resp = resp
		r.err = err
		close(r.done)
	})
}

// Response blocks until streaming completes and returns the accumulated response.
func (r *StreamResult) Response() (*Response, error) {
	<-r.done
	return r.resp, r.err
}

// Stream returns an iter.Seq2 that yields chunks as they arrive.
// The returned StreamResult provides the accumulated *Response after iteration.
// If the provider does not implement StreamingProvider, falls back to Chat.
func Stream(ctx context.Context, p LLMProvider, msgs []Message, tools []Tool) (iter.Seq2[StreamChunk, error], *StreamResult) {
	result := &StreamResult{done: make(chan struct{})}

	sp, ok := p.(StreamingProvider)
	if !ok {
		seq := func(yield func(StreamChunk, error) bool) {
			resp, err := p.Chat(ctx, msgs, tools)
			if err != nil {
				yield(StreamChunk{}, err)
				result.set(nil, err)
				return
			}
			yield(StreamChunk{Text: resp.Content}, nil)
			result.set(resp, nil)
		}
		return seq, result
	}

	seq := func(yield func(StreamChunk, error) bool) {
		resp, err := sp.ChatStream(ctx, msgs, tools, func(chunk StreamChunk) error {
			if !yield(chunk, nil) {
				return errStreamBreak
			}
			return nil
		})
		if err != nil && !errors.Is(err, errStreamBreak) {
			yield(StreamChunk{}, err)
			result.set(nil, err)
			return
		}
		result.set(resp, nil)
	}

	return seq, result
}

// StreamWithChan starts streaming and returns a channel of chunks plus a
// function that blocks until done and returns the accumulated response.
// The channel is closed when streaming completes.
// If the provider does not implement StreamingProvider, falls back to Chat.
func StreamWithChan(ctx context.Context, p LLMProvider, msgs []Message, tools []Tool) (<-chan StreamChunk, func() (*Response, error)) {
	ch := make(chan StreamChunk, 8)
	result := &StreamResult{done: make(chan struct{})}

	go func() {
		defer close(ch)

		sp, ok := p.(StreamingProvider)
		if !ok {
			resp, err := p.Chat(ctx, msgs, tools)
			if err != nil {
				result.set(nil, err)
				return
			}
			select {
			case ch <- StreamChunk{Text: resp.Content}:
			case <-ctx.Done():
				result.set(nil, ctx.Err())
				return
			}
			result.set(resp, nil)
			return
		}

		resp, err := sp.ChatStream(ctx, msgs, tools, func(chunk StreamChunk) error {
			select {
			case ch <- chunk:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		result.set(resp, err)
	}()

	return ch, func() (*Response, error) {
		<-result.done
		return result.resp, result.err
	}
}

// extractPrompts reconstructs sys/user prompts from messages for debug capture.
func extractPrompts(messages []Message) (string, string) {
	var sysText, userText strings.Builder
	for _, m := range messages {
		switch m.Role {
		case RoleSystem:
			if sysText.Len() > 0 {
				sysText.WriteString("\n\n")
			}
			sysText.WriteString(m.Content)
		case RoleUser:
			if userText.Len() > 0 {
				userText.WriteString("\n\n")
			}
			if m.Content != "" {
				userText.WriteString(m.Content)
			}
			for _, pt := range m.Parts {
				if v, ok := pt.(TextPart); ok {
					userText.WriteString(v.Text)
				}
			}
		}
	}
	return sysText.String(), userText.String()
}
