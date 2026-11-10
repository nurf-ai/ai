package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// A reportedUsageTransport reads usage fields the OpenAI client's Usage type
// does not decode off the response body as the client consumes it, into a
// sink carried on the request context; emitUsage then meters with them:
//   - usage.cost: some gateways (OpenRouter) bill each request at a rate set
//     at runtime by the upstream they route to, and report the charge there;
//     it replaces the models.json estimate.
//   - usage.prompt_tokens_details.cache_write_tokens: OpenAI's count of
//     prompt tokens written to its cache, billed at the cache-write rate.

type reportedUsage struct {
	usd        float64
	ok         bool
	cacheWrite int
}

type reportedUsageKey struct{}

// withReportedUsage attaches an empty usage sink to ctx.
func withReportedUsage(ctx context.Context) context.Context {
	return context.WithValue(ctx, reportedUsageKey{}, &reportedUsage{})
}

// reportedCostFromCtx returns the cost the provider reported for the
// requests made with ctx, if any.
func reportedCostFromCtx(ctx context.Context) (float64, bool) {
	s, _ := ctx.Value(reportedUsageKey{}).(*reportedUsage)
	if s == nil || !s.ok {
		return 0, false
	}
	return s.usd, true
}

// reportedCacheWritesFromCtx returns the prompt tokens the provider reported
// written to its cache for the requests made with ctx (0 if none).
func reportedCacheWritesFromCtx(ctx context.Context) int {
	s, _ := ctx.Value(reportedUsageKey{}).(*reportedUsage)
	if s == nil {
		return 0
	}
	return s.cacheWrite
}

type reportedUsageTransport struct {
	base http.RoundTripper
}

func (t reportedUsageTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	resp, err := base.RoundTrip(req)
	if err != nil || resp.Body == nil {
		return resp, err
	}
	sink, _ := req.Context().Value(reportedUsageKey{}).(*reportedUsage)
	if sink == nil {
		return resp, nil
	}
	resp.Body = &usageSniffer{
		ReadCloser: resp.Body,
		sink:       sink,
		sse:        strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream"),
	}
	return resp, nil
}

// usageSniffer passes the body through untouched, parsing the usage fields
// from it: per "data:" line of an SSE stream, or from the whole JSON body
// once read or closed.
type usageSniffer struct {
	io.ReadCloser
	sink *reportedUsage
	sse  bool
	buf  bytes.Buffer
	done bool
}

func (s *usageSniffer) Read(p []byte) (int, error) {
	n, err := s.ReadCloser.Read(p)
	if !s.done {
		s.buf.Write(p[:n])
		if s.sse {
			s.scanLines()
		}
		if err != nil {
			s.finish()
		}
	}
	return n, err
}

func (s *usageSniffer) Close() error {
	s.finish()
	return s.ReadCloser.Close()
}

func (s *usageSniffer) scanLines() {
	for {
		i := bytes.IndexByte(s.buf.Bytes(), '\n')
		if i < 0 {
			return
		}
		line := bytes.TrimSpace(s.buf.Next(i + 1))
		if data, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			s.parse(bytes.TrimSpace(data))
		}
	}
}

func (s *usageSniffer) finish() {
	if s.done {
		return
	}
	s.done = true
	if s.sse {
		if data, ok := bytes.CutPrefix(bytes.TrimSpace(s.buf.Bytes()), []byte("data:")); ok {
			s.parse(bytes.TrimSpace(data))
		}
	} else {
		s.parse(bytes.TrimSpace(s.buf.Bytes()))
	}
	s.buf.Reset()
}

func (s *usageSniffer) parse(data []byte) {
	if !bytes.Contains(data, []byte(`"cost"`)) && !bytes.Contains(data, []byte(`"cache_write_tokens"`)) {
		return
	}
	var body struct {
		Usage *struct {
			Cost                *float64 `json:"cost"`
			PromptTokensDetails *struct {
				CacheWriteTokens int `json:"cache_write_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &body) != nil || body.Usage == nil {
		return
	}
	// summed: one call can span several requests (instructor retries)
	if body.Usage.Cost != nil {
		s.sink.usd += *body.Usage.Cost
		s.sink.ok = true
	}
	if d := body.Usage.PromptTokensDetails; d != nil {
		s.sink.cacheWrite += d.CacheWriteTokens
	}
}
