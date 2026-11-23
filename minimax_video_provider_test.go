package ai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestMinimaxResolution(t *testing.T) {
	cases := map[string]string{"": "", "768P": "768P", "768p": "768P", "1080p": "768P", "1080P": "768P", "720p": "480P", "1440p": "2K", "2K": "2K", "4k": "480P", "weird": "WEIRD"}
	for in, want := range cases {
		if got := minimaxResolution(in); got != want {
			t.Errorf("minimaxResolution(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMinimaxPoll_FailedTaskSurfacesError(t *testing.T) {
	orig := minimaxPollInterval
	minimaxPollInterval = time.Millisecond
	t.Cleanup(func() { minimaxPollInterval = orig })

	body := `{"task":{"id":"1","status":"failed","error":{"code":"2013","message":"image size 64x64, expected each side in [256, 5760]"}}}`
	p := newMinimaxVideoProvider("k", "")
	p.httpClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: http.Header{}}, nil
	})}

	_, err := p.poll(context.Background(), "1")
	if err == nil || !strings.Contains(err.Error(), "expected each side in [256, 5760]") || !strings.Contains(err.Error(), "2013") {
		t.Fatalf("want provider reason in error, got %v", err)
	}
}
