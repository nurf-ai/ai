package ai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchVideoBytes_SendsHeadersAndBounds(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-goog-api-key") != "k" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte("mp4bytes"))
	}))
	defer srv.Close()

	if _, err := fetchVideoBytes(context.Background(), srv.URL, nil); err == nil {
		t.Fatal("expected 403 without the key header")
	}
	data, err := fetchVideoBytes(context.Background(), srv.URL, map[string]string{"x-goog-api-key": "k"})
	if err != nil || string(data) != "mp4bytes" {
		t.Fatalf("got %q err=%v", data, err)
	}
}
