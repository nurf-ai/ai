package ai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAICompatibleChat_HonorsMaxTokens(t *testing.T) {
	for _, provider := range []struct {
		name string
		make func(string) LLMProvider
	}{
		{name: "openrouter", make: func(baseURL string) LLMProvider {
			return newOpenRouterProvider("test", "moonshotai/kimi-k3", baseURL)
		}},
		{name: "huggingface", make: func(baseURL string) LLMProvider {
			return NewHuggingFaceProvider("test", "meta-llama/Llama-3.3-70B-Instruct", baseURL)
		}},
	} {
		t.Run(provider.name, func(t *testing.T) {
			got := make(chan int, 1)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					MaxTokens int  `json:"max_completion_tokens"`
					Stream    bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					http.Error(w, "invalid request", http.StatusBadRequest)
					return
				}
				got <- request.MaxTokens
				if request.Stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprint(w, "data: {\"id\":\"greet\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"id":"greet","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}]}`)
			}))
			t.Cleanup(srv.Close)
			p := provider.make(srv.URL)
			for _, stream := range []bool{false, true} {
				for _, limit := range []int{0, 8192, 128000} {
					t.Run(fmt.Sprintf("stream=%v/limit=%d", stream, limit), func(t *testing.T) {
						ctx := t.Context()
						if limit != 0 {
							ctx = WithMaxTokens(ctx, limit)
						}
						messages := []Message{{Role: RoleUser, Content: "greet"}}
						var err error
						if stream {
							_, err = p.(StreamingProvider).ChatStream(ctx, messages, nil, func(StreamChunk) error { return nil })
						} else {
							_, err = p.Chat(ctx, messages, nil)
						}
						if err != nil {
							t.Fatal(err)
						}
						want := limit
						if want == 0 {
							want = 4096
						}
						if value := <-got; value != want {
							t.Errorf("max_completion_tokens = %d, want %d", value, want)
						}
					})
				}
			}
		})
	}
}
