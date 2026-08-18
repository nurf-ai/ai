package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// openaiStreamLoop handles the SSE streaming loop for any provider using the
// go-openai SDK (OpenAI, Ollama, HuggingFace).
func openaiStreamLoop(ctx context.Context, raw *openai.Client, req openai.ChatCompletionRequest, cb func(StreamChunk) error) (*Response, openai.Usage, error) {
	req.Stream = true
	req.StreamOptions = &openai.StreamOptions{IncludeUsage: true}

	stream, err := raw.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, openai.Usage{}, err
	}
	defer stream.Close() //nolint:errcheck // fire-and-forget cleanup

	var content strings.Builder
	var toolCalls []ToolCall
	toolArgsBuf := map[int]*strings.Builder{}
	var usage openai.Usage

	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, usage, err
		}

		if chunk.Usage != nil {
			usage = *chunk.Usage
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			content.WriteString(delta.Content)
			if err := cb(StreamChunk{Text: delta.Content}); err != nil {
				resp := &Response{Content: content.String()}
				finalizeOpenAIToolCalls(&toolCalls, toolArgsBuf)
				if len(toolCalls) > 0 {
					resp.ToolCalls = toolCalls
				}
				return resp, usage, err
			}
		}

		for _, tc := range delta.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			for len(toolCalls) <= idx {
				toolCalls = append(toolCalls, ToolCall{})
				toolArgsBuf[len(toolCalls)-1] = &strings.Builder{}
			}
			if tc.ID != "" {
				toolCalls[idx].ID = tc.ID
			}
			if tc.Function.Name != "" {
				toolCalls[idx].Name = tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				toolArgsBuf[idx].WriteString(tc.Function.Arguments)
			}
		}
	}

	finalizeOpenAIToolCalls(&toolCalls, toolArgsBuf)
	resp := &Response{Content: content.String()}
	if len(toolCalls) > 0 {
		resp.ToolCalls = toolCalls
	}
	return resp, usage, nil
}

func finalizeOpenAIToolCalls(toolCalls *[]ToolCall, argsBuf map[int]*strings.Builder) {
	for idx, buf := range argsBuf {
		if idx < len(*toolCalls) && buf.Len() > 0 {
			(*toolCalls)[idx].Arguments = json.RawMessage(buf.String())
		}
	}
}
