package ai

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func elevenLabsServer(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("xi-api-key") != "k" {
			t.Errorf("xi-api-key = %q", r.Header.Get("xi-api-key"))
		}
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestElevenLabsTTS(t *testing.T) {
	var gotPath, gotQuery string
	var gotBody map[string]any
	srv := elevenLabsServer(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotQuery = r.URL.Path, r.URL.RawQuery
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("ID3audio"))
	})
	hook, evs := meterSpy()
	p := NewElevenLabsTTSProvider("k", "")
	p.client.baseURL = srv.URL
	p.SetMeter(hook)
	rc, err := p.Synthesize(context.Background(), TTSRequest{Text: "héllo", Speed: 1.1})
	if err != nil {
		t.Fatal(err)
	}
	audio, _ := io.ReadAll(rc)
	_ = rc.Close()
	if string(audio) != "ID3audio" {
		t.Errorf("audio = %q", audio)
	}
	if gotPath != "/v1/text-to-speech/"+DefaultElevenLabsVoice || gotQuery != "output_format=mp3_44100_128" {
		t.Errorf("request = %s?%s", gotPath, gotQuery)
	}
	if gotBody["model_id"] != DefaultElevenLabsTTSModel || gotBody["text"] != "héllo" {
		t.Errorf("body = %v", gotBody)
	}
	if vs, _ := gotBody["voice_settings"].(map[string]any); vs["speed"] != 1.1 {
		t.Errorf("voice_settings = %v", gotBody["voice_settings"])
	}
	if len(*evs) != 1 {
		t.Fatalf("meter events = %d", len(*evs))
	}
	// 5 runes at $40/M chars
	if ev := (*evs)[0]; ev.Provider != "elevenlabs" || ev.InputTokens != 5 || math.Abs(ev.EstimatedCostUSD-0.0002) > 1e-12 {
		t.Errorf("event = %+v", ev)
	}
}

func TestElevenLabsTTS_Errors(t *testing.T) {
	srv := elevenLabsServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"detail":{"type":"not_found","code":"voice_not_found","message":"A voice with voice_id 'nope' was not found."}}`))
	})
	p := NewElevenLabsTTSProvider("k", "")
	p.client.baseURL = srv.URL
	_, err := p.Synthesize(context.Background(), TTSRequest{Text: "hi", Voice: "nope"})
	if err == nil || !strings.Contains(err.Error(), "voice_not_found") || !strings.Contains(err.Error(), "404") {
		t.Errorf("err = %v", err)
	}
	if _, err := p.Synthesize(context.Background(), TTSRequest{Text: "hi", Format: "aac"}); err == nil {
		t.Error("expected unsupported format error")
	}
	if _, err := p.Synthesize(context.Background(), TTSRequest{}); err == nil {
		t.Error("expected empty input error")
	}
}

func TestElevenLabsSTT(t *testing.T) {
	srv := elevenLabsServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/speech-to-text" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if got := r.FormValue("model_id"); got != "scribe_v2" {
			t.Errorf("model_id = %q", got)
		}
		f, h, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(f)
		if h.Filename != "clip.mp3" || string(b) != "audio" {
			t.Errorf("file = %s %q", h.Filename, b)
		}
		_, _ = w.Write([]byte(`{"language_code":"eng","text":"hello there","audio_duration_secs":60}`))
	})
	hook, evs := meterSpy()
	p := NewElevenLabsSTTProvider("k", "")
	p.client.baseURL = srv.URL
	p.SetMeter(hook)
	text, err := p.Transcribe(context.Background(), strings.NewReader("audio"), "clip.mp3")
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello there" {
		t.Errorf("text = %q", text)
	}
	if len(*evs) != 1 {
		t.Fatalf("meter events = %d", len(*evs))
	}
	// one minute of scribe_v2 at $0.22/h
	if ev := (*evs)[0]; ev.Operation != "stt" || math.Abs(ev.EstimatedCostUSD-0.0036667) > 1e-9 || ev.Metadata["seconds"] != 60.0 {
		t.Errorf("event = %+v", ev)
	}
}

func TestElevenLabsAudio(t *testing.T) {
	var gotBody map[string]any
	audio := make([]byte, 32000) // 2 s at 128 kbps
	srv := elevenLabsServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/sound-generation" || r.URL.Query().Get("output_format") != "mp3_44100_128" {
			t.Errorf("request = %s", r.URL)
		}
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write(audio)
	})
	hook, evs := meterSpy()
	p := NewElevenLabsAudioProvider("k", "")
	p.client.baseURL = srv.URL
	p.SetMeter(hook)

	res, err := p.Generate(context.Background(), AudioRequest{Prompt: "door knock"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := gotBody["duration_seconds"]; ok {
		t.Errorf("duration sent unasked: %v", gotBody)
	}
	if !strings.HasPrefix(res.URL, "data:audio/mpeg;base64,") || res.FileSize != 32000 {
		t.Errorf("result = %+v", res)
	}
	if res.Duration != 2 || math.Abs(res.CostUSD-0.004) > 1e-12 {
		t.Errorf("duration %v cost %v, want 2s $0.004", res.Duration, res.CostUSD)
	}
	if len(*evs) != 1 || (*evs)[0].Metadata["estimated"] != true {
		t.Errorf("events = %+v", *evs)
	}

	res, err = p.Generate(context.Background(), AudioRequest{Prompt: "door knock", Duration: 5})
	if err != nil {
		t.Fatal(err)
	}
	if gotBody["duration_seconds"] != 5.0 || res.Duration != 5 || math.Abs(res.CostUSD-0.01) > 1e-12 {
		t.Errorf("body %v, result %+v", gotBody, res)
	}
}

func TestElevenLabsOutputFormat(t *testing.T) {
	for in, want := range map[string]string{"": "mp3_44100_128", "mp3": "mp3_44100_128", "pcm": "pcm_24000", "opus": "opus_48000_128", "wav": "wav_44100", "ulaw_8000": "ulaw_8000"} {
		if got, err := elevenLabsOutputFormat(in); err != nil || got != want {
			t.Errorf("%q → %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"aac", "flac"} {
		if _, err := elevenLabsOutputFormat(in); err == nil {
			t.Errorf("%q accepted", in)
		}
	}
	if got := elevenLabsAudioSeconds("pcm_24000", 48000); got != 1 {
		t.Errorf("pcm seconds = %v", got)
	}
	if got := elevenLabsAudioSeconds("ulaw_8000", 8000); got != 0 {
		t.Errorf("ulaw seconds = %v, want 0 (unknown)", got)
	}
}
