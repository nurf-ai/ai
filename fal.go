package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"
)

// DefaultFalQueueBase is the fal queue API root. Every fal model endpoint is
// addressed as {base}/{model_id}; the queue returns absolute status/result
// URLs which the client follows verbatim.
const DefaultFalQueueBase = "https://queue.fal.run"

const (
	falStatusInQueue    = "IN_QUEUE"
	falStatusInProgress = "IN_PROGRESS"
	falStatusCompleted  = "COMPLETED"

	falMaxPollInterval = 5 * time.Second
)

// FalClient is a minimal client for fal.ai's HTTP queue API
// (submit → poll status → fetch result). It is model-agnostic: any fal
// endpoint that takes a JSON input and returns a JSON output can be driven
// through Run.
type FalClient struct {
	apiKey       string
	http         *http.Client
	queueBase    string
	pollInterval time.Duration
}

// FalOption configures a FalClient.
type FalOption func(*FalClient)

// WithFalHTTPClient overrides the HTTP client (timeouts, transport, tests).
func WithFalHTTPClient(h *http.Client) FalOption {
	return func(c *FalClient) {
		if h != nil {
			c.http = h
		}
	}
}

// WithFalQueueBase overrides the queue API root (tests, proxies).
func WithFalQueueBase(base string) FalOption {
	return func(c *FalClient) {
		if base != "" {
			c.queueBase = strings.TrimRight(base, "/")
		}
	}
}

// WithFalPollInterval sets the initial status poll interval. It backs off
// geometrically up to 5s.
func WithFalPollInterval(d time.Duration) FalOption {
	return func(c *FalClient) {
		if d > 0 {
			c.pollInterval = d
		}
	}
}

// NewFalClient creates a fal queue client authenticated with apiKey.
func NewFalClient(apiKey string, opts ...FalOption) *FalClient {
	c := &FalClient{
		apiKey:       apiKey,
		http:         &http.Client{Timeout: 60 * time.Second},
		queueBase:    DefaultFalQueueBase,
		pollInterval: time.Second,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// FalRequest is the queue ticket returned by Submit.
type FalRequest struct {
	Endpoint      string `json:"-"`
	RequestID     string `json:"request_id"`
	StatusURL     string `json:"status_url"`
	ResponseURL   string `json:"response_url"`
	CancelURL     string `json:"cancel_url"`
	QueuePosition int    `json:"queue_position"`
}

// FalStatus is one status poll.
type FalStatus struct {
	Status        string `json:"status"`
	QueuePosition int    `json:"queue_position"`
	Metrics       struct {
		InferenceTime float64 `json:"inference_time"`
	} `json:"metrics"`
}

// Done reports whether the request has finished (success or failure — the
// result fetch tells which).
func (s *FalStatus) Done() bool { return s != nil && s.Status == falStatusCompleted }

// FalError is a non-2xx response from any fal endpoint.
type FalError struct {
	Status   int
	Endpoint string
	Message  string
}

func (e *FalError) Error() string {
	return fmt.Sprintf("fal %s: status %d: %s", e.Endpoint, e.Status, e.Message)
}

// Submit enqueues input on endpoint (e.g. "fal-ai/ltx-2.3/image-to-video/fast").
func (c *FalClient) Submit(ctx context.Context, endpoint string, input any) (*FalRequest, error) {
	endpoint = strings.Trim(endpoint, "/")
	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("fal submit: encode input: %w", err)
	}
	var req FalRequest
	if err := c.do(ctx, http.MethodPost, c.queueBase+"/"+endpoint, body, endpoint, &req); err != nil {
		return nil, err
	}
	req.Endpoint = endpoint
	if req.RequestID == "" {
		return nil, &FalError{Status: http.StatusBadGateway, Endpoint: endpoint, Message: "submit returned no request_id"}
	}
	if req.StatusURL == "" {
		req.StatusURL = c.queueBase + "/" + endpoint + "/requests/" + req.RequestID + "/status"
	}
	if req.ResponseURL == "" {
		req.ResponseURL = c.queueBase + "/" + endpoint + "/requests/" + req.RequestID
	}
	logger.Debug("fal submitted", zap.String("endpoint", endpoint), zap.String("request_id", req.RequestID), zap.Int("queue_position", req.QueuePosition))
	return &req, nil
}

// Status polls the request once.
func (c *FalClient) Status(ctx context.Context, req *FalRequest) (*FalStatus, error) {
	var st FalStatus
	if err := c.do(ctx, http.MethodGet, req.StatusURL, nil, req.Endpoint, &st); err != nil {
		return nil, err
	}
	return &st, nil
}

// Result fetches the model output. fal answers with a non-2xx status and a
// `detail` body when the run failed; that surfaces as *FalError.
func (c *FalClient) Result(ctx context.Context, req *FalRequest) (json.RawMessage, error) {
	var raw json.RawMessage
	if err := c.do(ctx, http.MethodGet, req.ResponseURL, nil, req.Endpoint, &raw); err != nil {
		return nil, err
	}
	return raw, nil
}

// Cancel asks the queue to drop a request that has not started yet.
func (c *FalClient) Cancel(ctx context.Context, req *FalRequest) error {
	if req.CancelURL == "" {
		return nil
	}
	return c.do(ctx, http.MethodPut, req.CancelURL, nil, req.Endpoint, nil)
}

// Wait polls until the request completes or ctx is done.
func (c *FalClient) Wait(ctx context.Context, req *FalRequest) (*FalStatus, error) {
	interval := c.pollInterval
	for {
		st, err := c.Status(ctx, req)
		if err != nil {
			return nil, err
		}
		if st.Done() {
			return st, nil
		}
		logger.Log(traceLevel, "fal waiting", zap.String("request_id", req.RequestID), zap.String("status", st.Status), zap.Int("queue_position", st.QueuePosition))
		select {
		case <-ctx.Done():
			// Best effort: free the runner slot if we bail while queued.
			cctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			_ = c.Cancel(cctx, req)
			cancel()
			return nil, ctx.Err()
		case <-time.After(interval):
		}
		interval = interval * 3 / 2
		if interval > falMaxPollInterval {
			interval = falMaxPollInterval
		}
	}
}

// Run submits input, waits for completion and returns the raw output JSON.
func (c *FalClient) Run(ctx context.Context, endpoint string, input any) (json.RawMessage, error) {
	req, err := c.Submit(ctx, endpoint, input)
	if err != nil {
		return nil, err
	}
	if _, err := c.Wait(ctx, req); err != nil {
		return nil, err
	}
	return c.Result(ctx, req)
}

func (c *FalClient) do(ctx context.Context, method, url string, body []byte, endpoint string, out any) error {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return fmt.Errorf("fal %s: %w", endpoint, err)
	}
	req.Header.Set("Authorization", "Key "+c.apiKey)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("fal %s: %w", endpoint, err)
	}
	defer resp.Body.Close() //nolint:errcheck
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return fmt.Errorf("fal %s: read body: %w", endpoint, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &FalError{Status: resp.StatusCode, Endpoint: endpoint, Message: falErrorMessage(data)}
	}
	if out == nil || len(data) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("fal %s: decode response: %w", endpoint, err)
	}
	return nil
}

// falErrorMessage flattens fal's error body — `detail` is either a string or
// a list of {loc, msg, type} validation errors.
func falErrorMessage(data []byte) string {
	var env struct {
		Detail json.RawMessage `json:"detail"`
		Error  string          `json:"error"`
	}
	if json.Unmarshal(data, &env) == nil {
		if env.Error != "" {
			return env.Error
		}
		var s string
		if json.Unmarshal(env.Detail, &s) == nil && s != "" {
			return s
		}
		var items []struct {
			Loc []any  `json:"loc"`
			Msg string `json:"msg"`
		}
		if json.Unmarshal(env.Detail, &items) == nil && len(items) > 0 {
			parts := make([]string, 0, len(items))
			for _, it := range items {
				loc := make([]string, 0, len(it.Loc))
				for _, l := range it.Loc {
					loc = append(loc, fmt.Sprint(l))
				}
				if len(loc) > 0 {
					parts = append(parts, strings.Join(loc, ".")+": "+it.Msg)
				} else {
					parts = append(parts, it.Msg)
				}
			}
			return strings.Join(parts, "; ")
		}
	}
	msg := strings.TrimSpace(string(data))
	if msg == "" {
		return "no error detail"
	}
	return truncate(msg, 300)
}

// IsFalError reports whether err wraps a *FalError.
func IsFalError(err error) bool {
	var fe *FalError
	return errors.As(err, &fe)
}
