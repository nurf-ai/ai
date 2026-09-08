package ai

import (
	"context"
	"encoding/json"
	"fmt"
)

// Realtime event types emitted on RealtimeProvider.Recv().
const (
	RTAudioDelta    = "audio_delta"
	RTAudioDone     = "audio_done"
	RTTextDelta     = "text_delta"
	RTTextDone      = "text_done"
	RTTranscript    = "transcript"
	RTToolCall      = "tool_call"
	RTSpeechStarted = "speech_started"
	RTSpeechStopped = "speech_stopped"
	RTResponseDone  = "response_done"
	RTError         = "error"
)

// RealtimeEvent is a decoded server event from a realtime session.
type RealtimeEvent struct {
	Type       string
	Audio      []byte
	Text       string
	ToolCall   *RealtimeToolCall
	Usage      *RealtimeUsage
	ResponseID string
	ItemID     string
	Error      error
}

// RealtimeToolCall is a completed function call from the model.
type RealtimeToolCall struct {
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// RealtimeUsage carries token counts from a completed response.
type RealtimeUsage struct {
	InputTokens       int `json:"input_tokens"`
	OutputTokens      int `json:"output_tokens"`
	InputAudioTokens  int `json:"input_audio_tokens"`
	OutputAudioTokens int `json:"output_audio_tokens"`
	TotalTokens       int `json:"total_tokens"`
}

// RealtimeSessionConfig configures a realtime session.
type RealtimeSessionConfig struct {
	Model                   string
	Voice                   string
	Instructions            string
	Temperature             float64
	Tools                   []Tool
	InputAudioFormat        string // pcm16 (default), g711_ulaw, g711_alaw
	OutputAudioFormat       string
	TurnDetection           *RealtimeTurnDetection
	InputAudioTranscription *RealtimeTranscriptionConfig
}

// RealtimeTurnDetection configures server-side VAD.
type RealtimeTurnDetection struct {
	Type              string  // "server_vad" or empty (= server_vad)
	Threshold         float64 // 0.0–1.0
	PrefixPaddingMs   int
	SilenceDurationMs int
}

// RealtimeTranscriptionConfig configures input audio transcription.
type RealtimeTranscriptionConfig struct {
	Model string // e.g. "gpt-4o-transcribe"
}

// RealtimeProvider is a bidirectional voice+text conversational session.
type RealtimeProvider interface {
	Connect(ctx context.Context, cfg RealtimeSessionConfig) error
	SendAudio(data []byte) error
	CommitAudio() error
	ClearAudio() error
	AddMessage(role, text string) error
	SendToolResult(callID, output string) error
	CreateResponse() error
	CancelResponse() error
	Recv() <-chan RealtimeEvent
	Close() error
}

// RealtimeMeterable is implemented by realtime providers that accept a meter hook.
type RealtimeMeterable interface {
	SetMeter(MeterHook)
}

// SetRealtimeMeter attaches a meter hook to any RealtimeProvider that
// satisfies RealtimeMeterable.
func SetRealtimeMeter(p RealtimeProvider, hook MeterHook) {
	if hook == nil || p == nil {
		return
	}
	if m, ok := p.(RealtimeMeterable); ok {
		m.SetMeter(hook)
	}
}

// NewRealtimeProvider creates a RealtimeProvider from a provider name + API key + model.
func NewRealtimeProvider(providerName, apiKey, model string) (RealtimeProvider, error) {
	switch providerName {
	case "openai":
		return NewOpenAIRealtimeProvider(apiKey, model), nil
	default:
		return nil, fmt.Errorf("unsupported realtime provider: %s", providerName)
	}
}
