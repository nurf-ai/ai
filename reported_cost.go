package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

// Some gateways (OpenRouter) bill each request at a rate set at runtime by
// the upstream they route to, and report the charge in usage.cost. A
// reportedCostTransport reads it off the response body as the OpenAI client
// consumes it, into a sink carried on the request context; emitUsage then
// records the reported cost instead of a models.json estimate.

type reportedCost struct {
	usd float64
	ok  bool
}

type reportedCostKey struct{}

// withReportedCost attaches an empty cost sink to ctx.
func withReportedCost(ctx context.Context) context.Context {
	return context.WithValue(ctx, reportedCostKey{}, &reportedCost{})
}

// reportedCostFromCtx returns the cost the provider reported for the call
// made with ctx, if any.
func reportedCostFromCtx(ctx context.Context) (float64, bool) {
	s, _ := ctx.Value(reportedCostKey{}).(*reportedCost)
	if s == nil || !s.ok {
		return 0, false
	}
	return s.usd, true
}

type reportedCostTransport struct {
	base http.RoundTripper
}

func (t reportedCostTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	resp, err := base.RoundTrip(req)
	if err != nil || resp.Body == nil {
		return resp, err
	}
	sink, _ := req.Context().Value(reportedCostKey{}).(*reportedCost)
	if sink == nil {
		return resp, nil
	}
	resp.Body = &costSniffer{
		ReadCloser: resp.Body,
		sink:       sink,
		sse:        strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream"),
	}
	return resp, nil
}

// costSniffer passes the body through untouched, parsing usage.cost from
// it: per "data:" line of an SSE stream, or from the whole JSON body once
// read or closed.
type costSniffer struct {
	io.ReadCloser
	sink *reportedCost
	sse  bool
	buf  bytes.Buffer
	done bool
}

func (s *costSniffer) Read(p []byte) (int, error) {
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

func (s *costSniffer) Close() error {
	s.finish()
	return s.ReadCloser.Close()
}

func (s *costSniffer) scanLines() {
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

func (s *costSniffer) finish() {
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

func (s *costSniffer) parse(data []byte) {
	if !bytes.Contains(data, []byte(`"cost"`)) {
		return
	}
	var body struct {
		Usage *struct {
			Cost *float64 `json:"cost"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &body) != nil || body.Usage == nil || body.Usage.Cost == nil {
		return
	}
	s.sink.usd = *body.Usage.Cost
	s.sink.ok = true
}
