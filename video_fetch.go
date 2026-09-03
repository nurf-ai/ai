package ai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// maxVideoFetchBytes bounds a downloaded clip (a 1080p 10 s clip is ~10-30 MB).
const maxVideoFetchBytes int64 = 96 << 20

// fetchVideoBytes downloads a generated clip with extra request headers (API
// keys) so callers that only hold the public-looking URL are not stuck with a
// 403. Bounded by maxVideoFetchBytes.
func fetchVideoBytes(ctx context.Context, url string, headers map[string]string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	client := &http.Client{Timeout: 3 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("fetch video: status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxVideoFetchBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxVideoFetchBytes {
		return nil, fmt.Errorf("fetch video: exceeds %d bytes", maxVideoFetchBytes)
	}
	return data, nil
}
