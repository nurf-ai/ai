package ai

import (
	"context"
	"strings"

	openai "github.com/sashabaranov/go-openai"
)

// A stream can end before the provider's usage report arrives: the callback
// stops it, the context is cancelled (a user hits stop, a tab closes), the
// connection breaks. The provider has still read the whole prompt and
// written what streamed, and bills both, so the stream reports them anyway,
// marked `partial` in the event metadata. Counts the provider never sent are
// estimated here and marked `estimated`.

// withUsageNote stamps how a stream's usage event was arrived at: partial
// (the stream was cut short), estimated (the provider sent no counts).
func withUsageNote(ctx context.Context, partial, estimated bool) context.Context {
	md := map[string]any{}
	if partial {
		md["partial"] = true
	}
	if estimated {
		md["estimated"] = true
	}
	if len(md) == 0 {
		return ctx
	}
	return WithMeterMetadata(ctx, md)
}

// promptTokens estimates the prompt a provider read: the text of every
// message and tool call. Images and tool schemas are not counted, so it
// errs low.
func promptTokens(messages []Message) int {
	var b strings.Builder
	for _, m := range messages {
		b.WriteString(m.Content)
		for _, p := range m.Parts {
			if t, ok := p.(TextPart); ok {
				b.WriteString(t.Text)
			}
		}
		for _, tc := range m.ToolCalls {
			b.WriteString(tc.Name)
			b.Write(tc.Arguments)
		}
	}
	return CountTokens(b.String())
}

// streamedTokens estimates the output a cut-short reply had streamed.
func streamedTokens(resp *Response) int {
	if resp == nil {
		return 0
	}
	var b strings.Builder
	b.WriteString(resp.Content)
	for _, tc := range resp.ToolCalls {
		b.WriteString(tc.Name)
		b.Write(tc.Arguments)
	}
	return CountTokens(b.String())
}

// openAIStreamUsage is the usage an OpenAI-compatible stream reports, or,
// when it ended without the usage chunk (only the last chunk carries one, and
// a cut-short stream never gets it), an estimate from the prompt and what
// streamed. cut says the stream ended early. ok is false when nothing is
// known to be billed: the request never got as far as output.
func openAIStreamUsage(ctx context.Context, messages []Message, resp *Response, usage openai.Usage, cut bool) (context.Context, openai.Usage, bool) {
	if usage.TotalTokens > 0 || usage.PromptTokens > 0 {
		return withUsageNote(ctx, cut, false), usage, true
	}
	if !streamedAnything(resp) {
		return ctx, usage, false
	}
	in, out := promptTokens(messages), streamedTokens(resp)
	return withUsageNote(ctx, cut, true), openai.Usage{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out}, true
}

// streamedAnything reports whether a cut-short reply got as far as output:
// a request refused before generating is not billed.
func streamedAnything(resp *Response) bool {
	return resp != nil && (resp.Content != "" || len(resp.ToolCalls) > 0)
}
