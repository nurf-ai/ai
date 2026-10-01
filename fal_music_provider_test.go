package ai

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

// Each endpoint's own limits are held before anything is sent, as InputErrors
// naming the field — in characters, as the endpoint counts them.
func TestFalMusicProvider_InputLimits(t *testing.T) {
	long := func(n int) string { return strings.Repeat("é", n) } // two bytes, one character
	for _, tc := range []struct {
		name  string
		req   MusicRequest
		field string
	}{
		{"empty prompt", MusicRequest{Prompt: "  "}, "prompt"},
		{"lyrics with instrumental", MusicRequest{Prompt: "x", Lyrics: "la", Instrumental: true}, "lyrics"},
		{"lyria: prompt and lyrics over 5000 together", MusicRequest{Prompt: long(2000), Lyrics: long(3000)}, "prompt"},
		{"minimax: prompt under 10", MusicRequest{Prompt: "lofi", Model: "fal-ai/minimax-music/v2.6"}, "prompt"},
		{"minimax: lyrics over 3500", MusicRequest{Prompt: "lofi hip hop beat", Lyrics: long(3501), Model: "fal-ai/minimax-music/v2.6"}, "lyrics"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestFalMusic(t, "", nil, nil)
			_, err := p.Generate(context.Background(), tc.req)
			var in *InputError
			if !errors.As(err, &in) || in.Field != tc.field {
				t.Fatalf("err = %v, want an InputError on %s", err, tc.field)
			}
		})
	}
	// at the limits, in characters, they pass (2,000 é is 4,000 bytes)
	if err := checkMusicInput("google/lyria-3.5", MusicRequest{Prompt: long(2000), Lyrics: long(2980)}); err != nil {
		t.Fatalf("within 5,000 characters: %v", err)
	}
	if err := checkMusicInput("fal-ai/minimax-music/v2.6", MusicRequest{Prompt: long(2000), Lyrics: long(3500)}); err != nil {
		t.Fatalf("minimax at its limits: %v", err)
	}
}

func newTestFalMusic(t *testing.T, model string, output any, onSubmit func(r *http.Request, body map[string]any)) *FalMusicProvider {
	t.Helper()
	srv, _ := newFalTestServer(t, output, onSubmit)
	t.Cleanup(srv.Close)
	return NewFalMusicProvider("test-key", model, WithFalQueueBase(srv.URL), WithFalPollInterval(time.Millisecond))
}

func TestFalMusicProvider_GenerateDefaultModel(t *testing.T) {
	var got map[string]any
	var gotPath string
	output := map[string]any{
		"audio":  map[string]any{"url": "https://v3.fal.media/files/x/output.mp3", "content_type": "audio/mpeg", "file_name": "output.mp3", "file_size": 4340730},
		"lyrics": "[[A0]]\n[[B1]]\n[:] Neon on the water\n[:] We ride the last train\n[[C2]]\n[:] Sing it loud",
	}
	p := newTestFalMusic(t, "", output, func(r *http.Request, body map[string]any) {
		got = body
		gotPath = r.URL.Path
	})
	var events []UsageEvent
	p.SetMeter(func(ev UsageEvent) { events = append(events, ev) })

	res, err := p.Generate(WithDebugSpanID(WithMeterOperation(context.Background(), "songs"), "span-1"), MusicRequest{
		Prompt: "80s synthpop, female vocals",
		Lyrics: "[Verse]\nNeon on the water",
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if gotPath != "/"+DefaultFalMusicModel {
		t.Fatalf("submitted to %q, want default model", gotPath)
	}
	if got["prompt"] != "80s synthpop, female vocals\n\nLyrics:\n[Verse]\nNeon on the water" {
		t.Errorf("lyrics not folded into the prompt: %q", got["prompt"])
	}
	if len(got) != 1 {
		t.Errorf("prompt-only endpoint got extra fields: %v", got)
	}
	if res.URL != "https://v3.fal.media/files/x/output.mp3" || res.ContentType != "audio/mpeg" || res.FileSize != 4340730 {
		t.Fatalf("unexpected result: %+v", res)
	}
	if want := "Neon on the water\nWe ride the last train\n\nSing it loud"; res.Lyrics != want {
		t.Errorf("lyrics = %q, want %q", res.Lyrics, want)
	}
	if res.CostUSD != 0.1 {
		t.Errorf("cost = %v, want 0.1 flat", res.CostUSD)
	}
	if len(events) != 1 || events[0].Provider != "fal" || events[0].Operation != "songs" ||
		events[0].Metadata["type"] != "music_gen" || events[0].DebugSpanID != "span-1" || events[0].EstimatedCostUSD != 0.1 {
		t.Fatalf("unexpected meter events: %+v", events)
	}
}

func TestFalMusicProvider_InstrumentalPrompt(t *testing.T) {
	var got map[string]any
	p := newTestFalMusic(t, "", map[string]any{"audio": map[string]any{"url": "u"}}, func(_ *http.Request, body map[string]any) { got = body })

	res, err := p.Generate(context.Background(), MusicRequest{Prompt: "lo-fi beat", Instrumental: true})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if got["prompt"] != "lo-fi beat\n\nInstrumental only, no vocals." {
		t.Errorf("prompt = %q", got["prompt"])
	}
	if res.ContentType != "audio/mpeg" {
		t.Errorf("content type default = %q", res.ContentType)
	}
	if res.Lyrics != "" {
		t.Errorf("lyrics = %q, want empty", res.Lyrics)
	}
}

func TestFalMusicProvider_MiniMaxFields(t *testing.T) {
	cases := []struct {
		name string
		req  MusicRequest
		want map[string]any
	}{
		{"lyrics", MusicRequest{Prompt: "indie folk", Lyrics: "[Chorus]\nla la"},
			map[string]any{"prompt": "indie folk", "is_instrumental": false, "lyrics": "[Chorus]\nla la"}},
		{"model writes the words", MusicRequest{Prompt: "indie folk"},
			map[string]any{"prompt": "indie folk", "is_instrumental": false, "lyrics_optimizer": true}},
		{"instrumental", MusicRequest{Prompt: "indie folk", Instrumental: true},
			map[string]any{"prompt": "indie folk", "is_instrumental": true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]any
			var gotPath string
			p := newTestFalMusic(t, "", map[string]any{"audio": map[string]any{"url": "u"}}, func(r *http.Request, body map[string]any) {
				got = body
				gotPath = r.URL.Path
			})
			tc.req.Model = "fal-ai/minimax-music/v2.6"
			res, err := p.Generate(context.Background(), tc.req)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			if gotPath != "/fal-ai/minimax-music/v2.6" {
				t.Errorf("model override not used: %q", gotPath)
			}
			if len(got) != len(tc.want) {
				t.Errorf("input = %v, want %v", got, tc.want)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("%s = %v, want %v", k, got[k], v)
				}
			}
			if res.Model != "fal-ai/minimax-music/v2.6" || res.CostUSD != 0.15 {
				t.Errorf("model/cost = %s/%v", res.Model, res.CostUSD)
			}
		})
	}
}

func TestFalMusicProvider_Errors(t *testing.T) {
	p := newTestFalMusic(t, "", nil, nil)
	if _, err := p.Generate(context.Background(), MusicRequest{}); err == nil {
		t.Error("empty prompt should fail")
	}
	if _, err := p.Generate(context.Background(), MusicRequest{Prompt: "x", Lyrics: "la", Instrumental: true}); err == nil {
		t.Error("lyrics + instrumental should fail")
	}
	if _, err := p.Generate(context.Background(), MusicRequest{Prompt: "x"}); err == nil {
		t.Error("runner error should surface")
	}

	empty := newTestFalMusic(t, "", map[string]any{"audio": map[string]any{}}, nil)
	if _, err := empty.Generate(context.Background(), MusicRequest{Prompt: "x"}); err == nil {
		t.Error("missing audio url should fail")
	}
}

func TestMusicPricing(t *testing.T) {
	if !IsMusicModel(DefaultFalMusicModel) {
		t.Fatalf("default music model %q is not priced", DefaultFalMusicModel)
	}
	if IsMusicModel("sonilo/v1.1/text-to-music") {
		t.Error("per-second audio model must not count as a per-track music model")
	}
	models := MusicModels()
	if len(models) < 2 || models[0] > models[1] {
		t.Errorf("MusicModels = %v, want ≥2 sorted ids", models)
	}
	if got := EstimateMusicCost("nope/unknown"); got != 0 {
		t.Errorf("unknown model cost = %v, want 0", got)
	}
}

func TestCleanLyrics(t *testing.T) {
	cases := map[string]string{
		"":                                     "",
		"plain line\nsecond":                   "plain line\nsecond",
		"[[A0]]\n[[B1]]\n[:] a\n[[C2]]\n[:] b": "a\n\nb",
		"[Verse]\nkept tag":                    "[Verse]\nkept tag",
		"[:] (Static in the crowd)":            "(Static in the crowd)",
	}
	for in, want := range cases {
		if got := cleanLyrics(in); got != want {
			t.Errorf("cleanLyrics(%q) = %q, want %q", in, got, want)
		}
	}
}
