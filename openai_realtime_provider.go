package ai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

const (
	openaiRealtimeURL     = "wss://api.openai.com/v1/realtime"
	openaiRealtimeDefault = "gpt-realtime-2"

	// rtPCMSampleRate is the sample rate the GA Realtime API expects for
	// raw 16-bit PCM (audio/pcm) in both directions.
	rtPCMSampleRate = 24000
)

// OpenAIRealtimeProvider implements RealtimeProvider over OpenAI's Realtime
// WebSocket API (wss://api.openai.com/v1/realtime).
type OpenAIRealtimeProvider struct {
	apiKey  string
	model   string
	meter   MeterHook
	baseURL string // override for testing

	conn      *websocket.Conn
	connCtx   context.Context
	events    chan RealtimeEvent
	outbox    chan json.RawMessage
	cancel    context.CancelFunc
	done      chan struct{}
	closeOnce sync.Once
}

func NewOpenAIRealtimeProvider(apiKey, model string) *OpenAIRealtimeProvider {
	if model == "" {
		model = openaiRealtimeDefault
	}
	return &OpenAIRealtimeProvider{apiKey: apiKey, model: model}
}

func (p *OpenAIRealtimeProvider) SetMeter(hook MeterHook) { p.meter = hook }

func (p *OpenAIRealtimeProvider) Connect(ctx context.Context, cfg RealtimeSessionConfig) error {
	model := cfg.Model
	if model == "" {
		model = p.model
	}

	base := p.baseURL
	if base == "" {
		base = openaiRealtimeURL
	}
	url := fmt.Sprintf("%s?model=%s", base, model)

	header := http.Header{
		"Authorization": []string{"Bearer " + p.apiKey},
	}

	conn, _, err := websocket.DefaultDialer.DialContext(ctx, url, header)
	if err != nil {
		return fmt.Errorf("realtime connect: %w", err)
	}

	connCtx, cancel := context.WithCancel(context.Background())
	p.conn = conn
	p.connCtx = connCtx
	p.events = make(chan RealtimeEvent, 64)
	p.outbox = make(chan json.RawMessage, 256)
	p.done = make(chan struct{})
	p.cancel = cancel

	go p.readLoop(connCtx)
	go p.writeLoop(connCtx)

	if err := p.sendSessionUpdate(cfg); err != nil {
		_ = p.Close()
		return err
	}
	return nil
}

func (p *OpenAIRealtimeProvider) Recv() <-chan RealtimeEvent { return p.events }

func (p *OpenAIRealtimeProvider) Close() error {
	p.closeOnce.Do(func() {
		if p.cancel != nil {
			p.cancel()
		}
		if p.conn != nil {
			_ = p.conn.Close()
		}
	})
	if p.done != nil {
		<-p.done
	}
	return nil
}

// --- send helpers ---

func (p *OpenAIRealtimeProvider) send(v any) error {
	if p.outbox == nil {
		return fmt.Errorf("realtime: not connected")
	}
	select {
	case <-p.connCtx.Done():
		return fmt.Errorf("realtime: connection closed")
	default:
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("realtime marshal: %w", err)
	}
	select {
	case p.outbox <- json.RawMessage(b):
		return nil
	default:
		return fmt.Errorf("realtime: send buffer full")
	}
}

type rtTypedEvent struct {
	Type string `json:"type"`
}

type rtAudioAppend struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

func (p *OpenAIRealtimeProvider) SendAudio(data []byte) error {
	return p.send(rtAudioAppend{
		Type:  "input_audio_buffer.append",
		Audio: base64.StdEncoding.EncodeToString(data),
	})
}

func (p *OpenAIRealtimeProvider) CommitAudio() error {
	return p.send(rtTypedEvent{Type: "input_audio_buffer.commit"})
}

func (p *OpenAIRealtimeProvider) ClearAudio() error {
	return p.send(rtTypedEvent{Type: "input_audio_buffer.clear"})
}

func (p *OpenAIRealtimeProvider) AddMessage(role, text string) error {
	return p.send(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type": "message",
			"role": role,
			"content": []map[string]any{
				{"type": "input_text", "text": text},
			},
		},
	})
}

func (p *OpenAIRealtimeProvider) SendToolResult(callID, output string) error {
	return p.send(map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	})
}

func (p *OpenAIRealtimeProvider) CreateResponse() error {
	return p.send(rtTypedEvent{Type: "response.create"})
}

func (p *OpenAIRealtimeProvider) CancelResponse() error {
	return p.send(rtTypedEvent{Type: "response.cancel"})
}

// --- session config ---

func (p *OpenAIRealtimeProvider) sendSessionUpdate(cfg RealtimeSessionConfig) error {
	session := map[string]any{"type": "realtime"}
	if cfg.Instructions != "" {
		session["instructions"] = cfg.Instructions
	}
	if len(cfg.Tools) > 0 {
		tools := make([]map[string]any, len(cfg.Tools))
		for i, t := range cfg.Tools {
			tools[i] = map[string]any{
				"type":        "function",
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			}
		}
		session["tools"] = tools
	}

	// GA API nests audio config under session.audio.{input,output}.
	audioInput := map[string]any{}
	audioOutput := map[string]any{}

	if f := rtAudioFormat(cfg.InputAudioFormat); f != nil {
		audioInput["format"] = f
	}
	if f := rtAudioFormat(cfg.OutputAudioFormat); f != nil {
		audioOutput["format"] = f
	}
	if cfg.Voice != "" {
		audioOutput["voice"] = cfg.Voice
	}
	if cfg.TurnDetection != nil {
		td := map[string]any{"type": cfg.TurnDetection.Type}
		if td["type"] == "" {
			td["type"] = "server_vad"
		}
		if cfg.TurnDetection.Threshold > 0 {
			td["threshold"] = cfg.TurnDetection.Threshold
		}
		if cfg.TurnDetection.PrefixPaddingMs > 0 {
			td["prefix_padding_ms"] = cfg.TurnDetection.PrefixPaddingMs
		}
		if cfg.TurnDetection.SilenceDurationMs > 0 {
			td["silence_duration_ms"] = cfg.TurnDetection.SilenceDurationMs
		}
		audioInput["turn_detection"] = td
	}
	if cfg.InputAudioTranscription != nil {
		audioInput["transcription"] = map[string]any{
			"model": cfg.InputAudioTranscription.Model,
		}
	}

	audio := map[string]any{}
	if len(audioInput) > 0 {
		audio["input"] = audioInput
	}
	if len(audioOutput) > 0 {
		audio["output"] = audioOutput
	}
	if len(audio) > 0 {
		session["audio"] = audio
	}

	return p.send(map[string]any{
		"type":    "session.update",
		"session": session,
	})
}

// rtAudioFormat maps a RealtimeSessionConfig audio format name onto the GA
// Realtime audio format object. The GA API only accepts the MIME-style names
// "audio/pcm", "audio/pcmu" and "audio/pcma" — the beta names ("pcm16",
// "g711_ulaw", "g711_alaw") are still accepted here as input. Returns nil for
// an empty name so the caller can leave the field off entirely.
func rtAudioFormat(name string) map[string]any {
	switch name {
	case "":
		return nil
	case "pcm16", "pcm", "audio/pcm":
		return map[string]any{"type": "audio/pcm", "rate": rtPCMSampleRate}
	case "g711_ulaw", "pcmu", "audio/pcmu":
		return map[string]any{"type": "audio/pcmu"}
	case "g711_alaw", "pcma", "audio/pcma":
		return map[string]any{"type": "audio/pcma"}
	default:
		return map[string]any{"type": name}
	}
}

// --- read/write loops ---

func (p *OpenAIRealtimeProvider) readLoop(ctx context.Context) {
	defer close(p.events)
	defer close(p.done)
	for {
		_, msg, err := p.conn.ReadMessage()
		if err != nil {
			if ctx.Err() == nil {
				select {
				case p.events <- RealtimeEvent{Type: RTError, Error: fmt.Errorf("realtime read: %w", err)}:
				default:
				}
			}
			return
		}
		var hdr struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(msg, &hdr) != nil {
			continue
		}
		ev := p.parseEvent(hdr.Type, msg)
		if ev == nil {
			continue
		}
		select {
		case p.events <- *ev:
		case <-ctx.Done():
			return
		}
	}
}

func (p *OpenAIRealtimeProvider) writeLoop(ctx context.Context) {
	for {
		select {
		case msg, ok := <-p.outbox:
			if !ok {
				return
			}
			if err := p.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				logger.Debug("realtime write", zap.Error(err))
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

// --- event parsing ---

func (p *OpenAIRealtimeProvider) parseEvent(typ string, raw []byte) *RealtimeEvent {
	switch typ {
	case "response.audio.delta":
		var ev struct {
			Delta      string `json:"delta"`
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		audio, err := base64.StdEncoding.DecodeString(ev.Delta)
		if err != nil {
			return nil
		}
		return &RealtimeEvent{Type: RTAudioDelta, Audio: audio, ResponseID: ev.ResponseID, ItemID: ev.ItemID}

	case "response.audio.done":
		var ev struct {
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{Type: RTAudioDone, ResponseID: ev.ResponseID, ItemID: ev.ItemID}

	case "response.text.delta":
		var ev struct {
			Delta      string `json:"delta"`
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{Type: RTTextDelta, Text: ev.Delta, ResponseID: ev.ResponseID, ItemID: ev.ItemID}

	case "response.text.done":
		var ev struct {
			Text       string `json:"text"`
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{Type: RTTextDone, Text: ev.Text, ResponseID: ev.ResponseID, ItemID: ev.ItemID}

	case "response.audio_transcript.delta":
		var ev struct {
			Delta      string `json:"delta"`
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{Type: RTTranscript, Text: ev.Delta, ResponseID: ev.ResponseID, ItemID: ev.ItemID}

	case "response.function_call_arguments.done":
		var ev struct {
			CallID     string `json:"call_id"`
			Name       string `json:"name"`
			Arguments  string `json:"arguments"`
			ResponseID string `json:"response_id"`
			ItemID     string `json:"item_id"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{
			Type: RTToolCall,
			ToolCall: &RealtimeToolCall{
				CallID:    ev.CallID,
				Name:      ev.Name,
				Arguments: json.RawMessage(ev.Arguments),
			},
			ResponseID: ev.ResponseID,
			ItemID:     ev.ItemID,
		}

	case "input_audio_buffer.speech_started":
		return &RealtimeEvent{Type: RTSpeechStarted}

	case "input_audio_buffer.speech_stopped":
		return &RealtimeEvent{Type: RTSpeechStopped}

	case "response.done":
		return p.parseResponseDone(raw)

	case "error":
		var ev struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(raw, &ev) != nil {
			return nil
		}
		return &RealtimeEvent{
			Type:  RTError,
			Error: fmt.Errorf("realtime %s: %s", ev.Error.Type, ev.Error.Message),
		}

	default:
		logger.Debug("realtime unhandled event", zap.String("type", typ))
		return nil
	}
}

func (p *OpenAIRealtimeProvider) parseResponseDone(raw []byte) *RealtimeEvent {
	var ev struct {
		ResponseID string `json:"response_id"`
		Response   struct {
			Usage struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
				InputTokenDetails struct {
					TextTokens    int `json:"text_tokens"`
					AudioTokens   int `json:"audio_tokens"`
					CachedTokens  int `json:"cached_tokens"`
				} `json:"input_token_details"`
				OutputTokenDetails struct {
					TextTokens  int `json:"text_tokens"`
					AudioTokens int `json:"audio_tokens"`
				} `json:"output_token_details"`
				TotalTokens int `json:"total_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if json.Unmarshal(raw, &ev) != nil {
		return nil
	}

	u := ev.Response.Usage
	usage := &RealtimeUsage{
		InputTokens:       u.InputTokens,
		OutputTokens:      u.OutputTokens,
		InputAudioTokens:  u.InputTokenDetails.AudioTokens,
		OutputAudioTokens: u.OutputTokenDetails.AudioTokens,
		TotalTokens:       u.TotalTokens,
	}

	if p.meter != nil {
		textIn := u.InputTokenDetails.TextTokens
		textOut := u.OutputTokenDetails.TextTokens
		audioIn := u.InputTokenDetails.AudioTokens
		audioOut := u.OutputTokenDetails.AudioTokens
		p.meter(UsageEvent{
			Provider:         "openai",
			Model:            p.model,
			Operation:        "realtime",
			InputTokens:      u.InputTokens,
			OutputTokens:     u.OutputTokens,
			TotalTokens:      u.TotalTokens,
			EstimatedCostUSD: EstimateRealtimeCost(p.model, textIn, textOut, audioIn, audioOut),
			Metadata: map[string]any{
				"type":              "realtime",
				"audio_input_tok":   audioIn,
				"audio_output_tok":  audioOut,
				"text_input_tok":    textIn,
				"text_output_tok":   textOut,
			},
		})
	}

	return &RealtimeEvent{Type: RTResponseDone, Usage: usage, ResponseID: ev.ResponseID}
}
