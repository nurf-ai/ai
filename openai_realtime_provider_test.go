package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func startRealtimeTestServer(t *testing.T, handler func(*websocket.Conn)) *httptest.Server {
	t.Helper()
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Logf("upgrade: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		handler(conn)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func wsURL(srv *httptest.Server) string {
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

func recvWithTimeout(t *testing.T, ch <-chan RealtimeEvent, d time.Duration) (RealtimeEvent, bool) {
	t.Helper()
	select {
	case ev, ok := <-ch:
		return ev, ok
	case <-time.After(d):
		t.Fatal("timeout waiting for event")
		return RealtimeEvent{}, false
	}
}

// writeAll sends multiple JSON values on a WebSocket, stopping at the first error.
func writeAll(conn *websocket.Conn, msgs ...any) error {
	for _, m := range msgs {
		if err := conn.WriteJSON(m); err != nil {
			return err
		}
	}
	return nil
}

func TestRealtimeConnect(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created", "session": map[string]any{}})

		_, msg, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var ev map[string]any
		_ = json.Unmarshal(msg, &ev)
		if ev["type"] == "session.update" {
			_ = conn.WriteJSON(map[string]any{"type": "session.updated"})
		}

		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)

	err := p.Connect(context.Background(), RealtimeSessionConfig{
		Voice:        "coral",
		Instructions: "you are helpful",
		Temperature:  0.8,
	})
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	if p.Recv() == nil {
		t.Fatal("Recv() channel is nil")
	}
}

func TestRealtimeTextResponse(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created"})

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var ev map[string]any
			_ = json.Unmarshal(msg, &ev)

			switch ev["type"] {
			case "session.update":
				_ = conn.WriteJSON(map[string]any{"type": "session.updated"})
			case "response.create":
				_ = writeAll(conn,
					map[string]any{
						"type": "response.text.delta", "delta": "Hello",
						"response_id": "r1", "item_id": "i1",
					},
					map[string]any{
						"type": "response.text.done", "text": "Hello world",
						"response_id": "r1", "item_id": "i1",
					},
					map[string]any{
						"type": "response.done", "response_id": "r1",
						"response": map[string]any{
							"usage": map[string]any{
								"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
								"input_token_details":  map[string]any{"text_tokens": 8, "audio_tokens": 2},
								"output_token_details": map[string]any{"text_tokens": 3, "audio_tokens": 2},
							},
						},
					},
				)
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	if err := p.CreateResponse(); err != nil {
		t.Fatalf("create response: %v", err)
	}

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTTextDelta || ev.Text != "Hello" {
		t.Fatalf("want text_delta 'Hello', got %s %q", ev.Type, ev.Text)
	}

	ev, _ = recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTTextDone || ev.Text != "Hello world" {
		t.Fatalf("want text_done 'Hello world', got %s %q", ev.Type, ev.Text)
	}

	ev, _ = recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTResponseDone {
		t.Fatalf("want response_done, got %s", ev.Type)
	}
	if ev.Usage == nil {
		t.Fatal("usage is nil")
	}
	if ev.Usage.TotalTokens != 15 {
		t.Fatalf("want 15 total tokens, got %d", ev.Usage.TotalTokens)
	}
}

func TestRealtimeAudioRoundtrip(t *testing.T) {
	audioPayload := []byte{0x01, 0x02, 0x03, 0x04}
	b64Audio := base64.StdEncoding.EncodeToString(audioPayload)

	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created"})

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var ev map[string]any
			_ = json.Unmarshal(msg, &ev)

			switch ev["type"] {
			case "input_audio_buffer.append":
				if ev["audio"] != b64Audio {
					return
				}
			case "input_audio_buffer.commit":
				_ = writeAll(conn,
					map[string]any{
						"type": "response.audio.delta", "delta": b64Audio,
						"response_id": "r1", "item_id": "i1",
					},
					map[string]any{
						"type": "response.audio.done",
						"response_id": "r1", "item_id": "i1",
					},
				)
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	if err := p.SendAudio(audioPayload); err != nil {
		t.Fatalf("send audio: %v", err)
	}
	if err := p.CommitAudio(); err != nil {
		t.Fatalf("commit: %v", err)
	}

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTAudioDelta {
		t.Fatalf("want audio_delta, got %s", ev.Type)
	}
	if len(ev.Audio) != len(audioPayload) {
		t.Fatalf("want %d audio bytes, got %d", len(audioPayload), len(ev.Audio))
	}

	ev, _ = recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTAudioDone {
		t.Fatalf("want audio_done, got %s", ev.Type)
	}
}

func TestRealtimeToolCall(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created"})

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var ev map[string]any
			_ = json.Unmarshal(msg, &ev)

			switch ev["type"] {
			case "response.create":
				_ = conn.WriteJSON(map[string]any{
					"type": "response.function_call_arguments.done",
					"call_id": "call_1", "name": "get_weather",
					"arguments":   `{"city":"SF"}`,
					"response_id": "r1", "item_id": "i1",
				})
			case "conversation.item.create":
				item, _ := ev["item"].(map[string]any)
				if item["type"] == "function_call_output" {
					_ = conn.WriteJSON(map[string]any{
						"type": "response.text.delta", "delta": "sunny",
						"response_id": "r2", "item_id": "i2",
					})
				}
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "get weather",
			Parameters:  map[string]any{"type": "object"},
		}},
	}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	if err := p.CreateResponse(); err != nil {
		t.Fatalf("create response: %v", err)
	}

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTToolCall {
		t.Fatalf("want tool_call, got %s", ev.Type)
	}
	if ev.ToolCall.Name != "get_weather" || ev.ToolCall.CallID != "call_1" {
		t.Fatalf("unexpected tool call: %+v", ev.ToolCall)
	}

	if err := p.SendToolResult("call_1", `{"weather":"sunny"}`); err != nil {
		t.Fatalf("send tool result: %v", err)
	}
	if err := p.CreateResponse(); err != nil {
		t.Fatalf("create response: %v", err)
	}

	ev, _ = recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTTextDelta || ev.Text != "sunny" {
		t.Fatalf("want text_delta 'sunny', got %s %q", ev.Type, ev.Text)
	}
}

func TestRealtimeMeter(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created"})

		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var ev map[string]any
			_ = json.Unmarshal(msg, &ev)
			if ev["type"] == "response.create" {
				_ = conn.WriteJSON(map[string]any{
					"type": "response.done", "response_id": "r1",
					"response": map[string]any{
						"usage": map[string]any{
							"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
							"input_token_details":  map[string]any{"text_tokens": 20, "audio_tokens": 80},
							"output_token_details": map[string]any{"text_tokens": 10, "audio_tokens": 40},
						},
					},
				})
			}
		}
	})

	var mu sync.Mutex
	var captured UsageEvent
	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	p.SetMeter(func(ev UsageEvent) {
		mu.Lock()
		captured = ev
		mu.Unlock()
	})

	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	if err := p.CreateResponse(); err != nil {
		t.Fatalf("create response: %v", err)
	}

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTResponseDone {
		t.Fatalf("want response_done, got %s", ev.Type)
	}

	mu.Lock()
	defer mu.Unlock()
	if captured.Provider != "openai" {
		t.Fatalf("want provider openai, got %s", captured.Provider)
	}
	if captured.TotalTokens != 150 {
		t.Fatalf("want 150 total tokens, got %d", captured.TotalTokens)
	}
	if captured.EstimatedCostUSD <= 0 {
		t.Fatal("expected non-zero cost estimate")
	}
}

func TestRealtimeVADEvents(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = writeAll(conn,
			map[string]any{"type": "session.created"},
			map[string]any{"type": "input_audio_buffer.speech_started", "audio_start_ms": 0},
			map[string]any{"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 1500},
		)
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTSpeechStarted {
		t.Fatalf("want speech_started, got %s", ev.Type)
	}
	ev, _ = recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTSpeechStopped {
		t.Fatalf("want speech_stopped, got %s", ev.Type)
	}
}

func TestRealtimeClose(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = conn.WriteJSON(map[string]any{"type": "session.created"})
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}

	if err := p.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	_, ok := <-p.Recv()
	if ok {
		t.Fatal("expected Recv channel to be closed")
	}

	if err := p.SendAudio([]byte{1}); err == nil {
		t.Fatal("expected error on send after close")
	}
}

func TestRealtimeErrorEvent(t *testing.T) {
	srv := startRealtimeTestServer(t, func(conn *websocket.Conn) {
		_ = writeAll(conn,
			map[string]any{"type": "session.created"},
			map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    "invalid_request_error",
					"message": "bad request",
				},
			},
		)
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	})

	p := NewOpenAIRealtimeProvider("test-key", "gpt-realtime-2")
	p.baseURL = wsURL(srv)
	if err := p.Connect(context.Background(), RealtimeSessionConfig{}); err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = p.Close() }()

	ev, _ := recvWithTimeout(t, p.Recv(), 2*time.Second)
	if ev.Type != RTError {
		t.Fatalf("want error event, got %s", ev.Type)
	}
	if ev.Error == nil || !strings.Contains(ev.Error.Error(), "bad request") {
		t.Fatalf("unexpected error: %v", ev.Error)
	}
}

func TestRealtimeNotConnected(t *testing.T) {
	p := NewOpenAIRealtimeProvider("key", "model")
	if err := p.SendAudio([]byte{1}); err == nil {
		t.Fatal("expected error when not connected")
	}
}

func TestRealtimePricingCoverage(t *testing.T) {
	for _, model := range []string{"gpt-realtime-2", "gpt-realtime-2-mini"} {
		if !IsRealtimeModel(model) {
			t.Errorf("%s not recognized as realtime model", model)
		}
		cost := EstimateRealtimeCost(model, 100, 50, 1000, 500)
		if cost <= 0 {
			t.Errorf("%s: expected positive cost, got %f", model, cost)
		}
	}
}
