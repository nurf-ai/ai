//go:build integration

package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/joho/godotenv"
)

func TestMain(m *testing.M) {
	godotenv.Overload(".env.test")
	os.Exit(m.Run())
}

type greeting struct {
	Message string `json:"message"`
}

func newCostTracker(t *testing.T) MeterHook {
	var mu sync.Mutex
	var totalUSD float64
	var totalIn, totalOut, totalCached int

	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		t.Logf("##cost## usd=%.6f in=%d out=%d cached=%d", totalUSD, totalIn, totalOut, totalCached)
	})

	return func(ev UsageEvent) {
		mu.Lock()
		totalUSD += ev.EstimatedCostUSD
		totalIn += ev.InputTokens
		totalOut += ev.OutputTokens
		totalCached += ev.CacheReadInputTokens
		mu.Unlock()
	}
}

func testPNGBytesSize(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{0, 0, 255, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode test png: %v", err)
	}
	return buf.Bytes()
}

func testPNGBytes(t *testing.T) []byte { return testPNGBytesSize(t, 64, 64) }

var testSchema = json.RawMessage(`{
	"type": "object",
	"properties": {
		"message": {"type": "string"}
	},
	"required": ["message"]
}`)

// --- Anthropic -----------------------------------------------------------

func TestAnthropicChat_Integration(t *testing.T) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("Chat", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		if err != nil {
			t.Fatalf("chat: %v", err)
		}
		if resp.Content == "" {
			t.Fatal("empty response")
		}
		t.Logf("response: %s", resp.Content)
	})

	t.Run("Reasoning", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		ctx := WithReasoningEffort(ctx, "low")
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what is 2+2?"}}, nil)
		if err != nil {
			t.Fatalf("reasoning: %v", err)
		}
		if resp.Content == "" {
			t.Fatal("empty response")
		}
		t.Logf("response: %s", resp.Content)
	})

	t.Run("StructuredOutput", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		var out greeting
		err := p.CreateStructuredOutput(ctx, "greet me briefly", "respond with a greeting", &out)
		if err != nil {
			t.Fatalf("structured output: %v", err)
		}
		if out.Message == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %s", out.Message)
	})

	t.Run("StructuredOutputFromSchema", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		result, err := p.CreateStructuredOutputFromSchema(ctx, "greet me briefly", "respond with a greeting", testSchema)
		if err != nil {
			t.Fatalf("structured output from schema: %v", err)
		}
		if result["message"] == nil || result["message"] == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %v", result["message"])
	})

	t.Run("ChatWithTools", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		tools := []Tool{{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
		}}
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what's the weather in tokyo?"}}, tools)
		if err != nil {
			t.Fatalf("chat with tools: %v", err)
		}
		if resp.Content == "" && len(resp.ToolCalls) == 0 {
			t.Fatal("no content or tool calls")
		}
		if len(resp.ToolCalls) > 0 {
			t.Logf("tool call: %s(%s)", resp.ToolCalls[0].Name, string(resp.ToolCalls[0].Arguments))
		} else {
			t.Logf("response: %s", resp.Content)
		}
	})

	t.Run("Stream", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		chunks, result := Stream(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		var got strings.Builder
		for chunk, err := range chunks {
			if err != nil {
				t.Fatalf("stream chunk error: %v", err)
			}
			got.WriteString(chunk.Text)
		}
		if got.Len() == 0 {
			t.Fatal("no streamed text")
		}
		t.Logf("streamed: %s", got.String())
		resp, err := result.Response()
		if err != nil {
			t.Fatalf("stream response: %v", err)
		}
		if resp.Content != got.String() {
			t.Errorf("response %q != streamed %q", resp.Content, got.String())
		}
	})

	t.Run("StreamWithChan", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")
		p.SetMeter(newCostTracker(t))
		ch, wait := StreamWithChan(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		var got strings.Builder
		for chunk := range ch {
			got.WriteString(chunk.Text)
		}
		if got.Len() == 0 {
			t.Fatal("no streamed text")
		}
		resp, err := wait()
		if err != nil {
			t.Fatalf("stream wait: %v", err)
		}
		if resp.Content != got.String() {
			t.Errorf("response %q != streamed %q", resp.Content, got.String())
		}
	})

	t.Run("PromptCaching", func(t *testing.T) {
		t.Parallel()
		p := NewAnthropicProvider(key, "claude-haiku-4-5")

		costHook := newCostTracker(t)
		var mu sync.Mutex
		var events []UsageEvent
		p.SetMeter(func(ev UsageEvent) {
			costHook(ev)
			mu.Lock()
			events = append(events, ev)
			mu.Unlock()
		})

		sysPrompt := "You are a helpful assistant.\n" + strings.Repeat("Context line for prompt cache integration test padding. ", 500)
		cacheCtx := WithCacheSysPrompt(ctx)
		msgs := []Message{
			{Role: RoleSystem, Content: sysPrompt},
			{Role: RoleUser, Content: "say hi in one word"},
		}

		if _, err := p.Chat(cacheCtx, msgs, nil); err != nil {
			t.Fatalf("first call: %v", err)
		}

		mu.Lock()
		if len(events) == 0 || events[0].CacheCreationInputTokens == 0 {
			mu.Unlock()
			t.Skip("prompt caching not available for this model/key (cache_create=0)")
		}
		mu.Unlock()

		if _, err := p.Chat(cacheCtx, msgs, nil); err != nil {
			t.Fatalf("second call: %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		if len(events) < 2 {
			t.Fatalf("expected 2 meter events, got %d", len(events))
		}
		if events[1].CacheReadInputTokens == 0 {
			t.Error("second call: expected CacheReadInputTokens > 0")
		}
		t.Logf("call 1: cache_create=%d cache_read=%d", events[0].CacheCreationInputTokens, events[0].CacheReadInputTokens)
		t.Logf("call 2: cache_create=%d cache_read=%d", events[1].CacheCreationInputTokens, events[1].CacheReadInputTokens)
	})
}

// --- OpenAI --------------------------------------------------------------

func TestOpenAI_Integration(t *testing.T) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("Chat", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-4o-mini")
		p.SetMeter(newCostTracker(t))
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		if err != nil {
			t.Fatalf("chat: %v", err)
		}
		if resp.Content == "" {
			t.Fatal("empty response")
		}
		t.Logf("response: %s", resp.Content)
	})

	t.Run("Reasoning", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-5-mini")
		p.SetMeter(newCostTracker(t))
		ctx := WithReasoningEffort(ctx, "low")
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what is 2+2?"}}, nil)
		if err != nil {
			t.Fatalf("reasoning: %v", err)
		}
		if resp.Content == "" {
			t.Fatal("empty response")
		}
		t.Logf("response: %s", resp.Content)
	})

	t.Run("StructuredOutput", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-4o-mini")
		p.SetMeter(newCostTracker(t))
		var out greeting
		err := p.CreateStructuredOutput(ctx, "greet me briefly", "respond with a greeting", &out)
		if err != nil {
			t.Fatalf("structured output: %v", err)
		}
		if out.Message == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %s", out.Message)
	})

	t.Run("StructuredOutputFromSchema", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-4o-mini")
		p.SetMeter(newCostTracker(t))
		result, err := p.CreateStructuredOutputFromSchema(ctx, "greet me briefly", "respond with a greeting", testSchema)
		if err != nil {
			t.Fatalf("structured output from schema: %v", err)
		}
		if result["message"] == nil || result["message"] == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %v", result["message"])
	})

	t.Run("ChatWithTools", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-4o-mini")
		p.SetMeter(newCostTracker(t))
		tools := []Tool{{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
		}}
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what's the weather in tokyo?"}}, tools)
		if err != nil {
			t.Fatalf("chat with tools: %v", err)
		}
		if resp.Content == "" && len(resp.ToolCalls) == 0 {
			t.Fatal("no content or tool calls")
		}
		if len(resp.ToolCalls) > 0 {
			t.Logf("tool call: %s(%s)", resp.ToolCalls[0].Name, string(resp.ToolCalls[0].Arguments))
		}
	})

	t.Run("Stream", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "gpt-4o-mini")
		p.SetMeter(newCostTracker(t))
		chunks, result := Stream(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		var got strings.Builder
		for chunk, err := range chunks {
			if err != nil {
				t.Fatalf("stream chunk error: %v", err)
			}
			got.WriteString(chunk.Text)
		}
		if got.Len() == 0 {
			t.Fatal("no streamed text")
		}
		t.Logf("streamed: %s", got.String())
		resp, err := result.Response()
		if err != nil {
			t.Fatalf("stream response: %v", err)
		}
		if resp.Content != got.String() {
			t.Errorf("response %q != streamed %q", resp.Content, got.String())
		}
	})

	t.Run("Embeddings", func(t *testing.T) {
		t.Parallel()
		p := NewOpenAIProvider(key, "")
		p.SetMeter(newCostTracker(t))
		vectors, err := p.EmbedText(ctx, []string{"hello world", "goodbye"})
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		if len(vectors) != 2 {
			t.Fatalf("expected 2 vectors, got %d", len(vectors))
		}
		if len(vectors[0]) == 0 {
			t.Fatal("empty embedding vector")
		}
		t.Logf("dims: %d", len(vectors[0]))
	})

	t.Run("STT", func(t *testing.T) {
		t.Parallel()
		stt := newOpenAISTTProvider(key, "whisper-1")
		// Minimal valid WAV: 44-byte header + 1 sample of silence
		wav := make([]byte, 46)
		copy(wav, "RIFF")
		wav[4], wav[5], wav[6], wav[7] = 38, 0, 0, 0
		copy(wav[8:], "WAVE")
		copy(wav[12:], "fmt ")
		wav[16], wav[17], wav[18], wav[19] = 16, 0, 0, 0
		wav[20], wav[21] = 1, 0
		wav[22], wav[23] = 1, 0
		wav[24], wav[25], wav[26], wav[27] = 0x80, 0x3E, 0, 0
		wav[28], wav[29], wav[30], wav[31] = 0x00, 0x7D, 0, 0
		wav[32], wav[33] = 2, 0
		wav[34], wav[35] = 16, 0
		copy(wav[36:], "data")
		wav[40], wav[41], wav[42], wav[43] = 2, 0, 0, 0
		_, err := stt.Transcribe(ctx, strings.NewReader(string(wav)), "silence.wav")
		if err != nil {
			t.Logf("stt (may be expected for silent audio): %v", err)
		} else {
			t.Log("stt: ok")
		}
	})

	t.Run("Moderation", func(t *testing.T) {
		t.Parallel()
		mod := NewOpenAIModerationProvider(key)
		result, err := mod.Check(ctx, "hello, how are you?")
		if err != nil {
			t.Fatalf("moderation: %v", err)
		}
		if result.Flagged {
			t.Fatal("benign input flagged")
		}
		t.Logf("flagged: %v", result.Flagged)
	})

	t.Run("ImageGenerate", func(t *testing.T) {
		t.Parallel()
		img := newOpenAIImageProvider(key, "")
		b64, err := img.Generate(ctx, "a solid blue square", "", "")
		if err != nil {
			t.Fatalf("image generate: %v", err)
		}
		if len(b64) == 0 {
			t.Fatal("empty image")
		}
		t.Logf("image b64 len: %d", len(b64))
	})

	t.Run("ImageEdit", func(t *testing.T) {
		t.Parallel()
		img := newOpenAIImageProvider(key, "")
		b64, err := img.Edit(ctx, testPNGBytes(t), "add a small red circle in the center")
		if err != nil {
			t.Fatalf("image edit: %v", err)
		}
		if len(b64) == 0 {
			t.Fatal("empty edited image")
		}
		t.Logf("edited image b64 len: %d", len(b64))
	})
}

// --- Gemini --------------------------------------------------------------

func TestGemini_Integration(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("Chat", func(t *testing.T) {
		t.Parallel()
		p, err := NewGeminiProvider(ctx, key, "gemini-3.6-flash")
		if err != nil {
			t.Fatalf("create provider: %v", err)
		}
		p.SetMeter(newCostTracker(t))
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		if err != nil {
			t.Fatalf("chat: %v", err)
		}
		if resp.Content == "" {
			t.Fatal("empty response")
		}
		t.Logf("response: %s", resp.Content)
	})

	t.Run("StructuredOutput", func(t *testing.T) {
		t.Parallel()
		p, err := NewGeminiProvider(ctx, key, "gemini-3.6-flash")
		if err != nil {
			t.Fatalf("create provider: %v", err)
		}
		p.SetMeter(newCostTracker(t))
		var out greeting
		err = p.CreateStructuredOutput(ctx, "greet me briefly", "respond with a greeting", &out)
		if err != nil {
			t.Fatalf("structured output: %v", err)
		}
		if out.Message == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %s", out.Message)
	})

	t.Run("StructuredOutputFromSchema", func(t *testing.T) {
		t.Parallel()
		p, err := NewGeminiProvider(ctx, key, "gemini-3.6-flash")
		if err != nil {
			t.Fatalf("create provider: %v", err)
		}
		p.SetMeter(newCostTracker(t))
		result, err := p.CreateStructuredOutputFromSchema(ctx, "greet me briefly", "respond with a greeting", testSchema)
		if err != nil {
			t.Fatalf("structured output from schema: %v", err)
		}
		if result["message"] == nil || result["message"] == "" {
			t.Fatal("empty message")
		}
		t.Logf("greeting: %v", result["message"])
	})

	t.Run("ChatWithTools", func(t *testing.T) {
		t.Parallel()
		p, err := NewGeminiProvider(ctx, key, "gemini-3.6-flash")
		if err != nil {
			t.Fatalf("create provider: %v", err)
		}
		p.SetMeter(newCostTracker(t))
		tools := []Tool{{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
		}}
		resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what's the weather in tokyo?"}}, tools)
		if err != nil {
			t.Fatalf("chat with tools: %v", err)
		}
		if resp.Content == "" && len(resp.ToolCalls) == 0 {
			t.Fatal("no content or tool calls")
		}
		if len(resp.ToolCalls) > 0 {
			t.Logf("tool call: %s(%s)", resp.ToolCalls[0].Name, string(resp.ToolCalls[0].Arguments))
		}
	})

	t.Run("Stream", func(t *testing.T) {
		t.Parallel()
		p, err := NewGeminiProvider(ctx, key, "gemini-3.6-flash")
		if err != nil {
			t.Fatalf("create provider: %v", err)
		}
		p.SetMeter(newCostTracker(t))
		chunks, result := Stream(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
		var got strings.Builder
		for chunk, err := range chunks {
			if err != nil {
				t.Fatalf("stream chunk error: %v", err)
			}
			got.WriteString(chunk.Text)
		}
		if got.Len() == 0 {
			t.Fatal("no streamed text")
		}
		t.Logf("streamed: %s", got.String())
		resp, err := result.Response()
		if err != nil {
			t.Fatalf("stream response: %v", err)
		}
		if resp.Content != got.String() {
			t.Errorf("response %q != streamed %q", resp.Content, got.String())
		}
	})

	t.Run("ImageGenerate", func(t *testing.T) {
		t.Parallel()
		img, err := newGeminiImageProvider(ctx, key, "")
		if err != nil {
			t.Fatalf("create image provider: %v", err)
		}
		b64, err := img.Generate(ctx, "a solid blue square", "", "")
		if err != nil {
			t.Fatalf("image generate: %v", err)
		}
		if len(b64) == 0 {
			t.Fatal("empty image")
		}
		t.Logf("image b64 len: %d", len(b64))
	})

	t.Run("ImageEdit", func(t *testing.T) {
		t.Parallel()
		img, err := newGeminiImageProvider(ctx, key, "")
		if err != nil {
			t.Fatalf("create image provider: %v", err)
		}
		b64, err := img.Edit(ctx, testPNGBytes(t), "add a small red circle in the center")
		if err != nil {
			t.Fatalf("image edit: %v", err)
		}
		if len(b64) == 0 {
			t.Fatal("empty edited image")
		}
		t.Logf("edited image b64 len: %d", len(b64))
	})

	t.Run("ImageEditWithReference", func(t *testing.T) {
		t.Parallel()
		img, err := newGeminiImageProvider(ctx, key, "")
		if err != nil {
			t.Fatalf("create image provider: %v", err)
		}
		source := testPNGBytes(t)
		b64, err := img.EditWithReference(ctx, source, source, "Generate a new image: take the first image and recolor it using the palette of the second image")
		if err != nil {
			t.Fatalf("image edit with reference: %v", err)
		}
		if len(b64) == 0 {
			t.Fatal("empty edited image")
		}
		t.Logf("edited image b64 len: %d", len(b64))
	})
}

// --- Hugging Face --------------------------------------------------------

func TestHuggingFace_Integration(t *testing.T) {
	key := os.Getenv("HF_API_KEY")
	if key == "" {
		t.Skip("HF_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()
	models := []string{
		"moonshotai/Kimi-K2-Instruct-0905",
		"moonshotai/Kimi-K3",
	}

	for _, model := range models {
		model := model
		t.Run(model, func(t *testing.T) {
			t.Parallel()

			t.Run("Chat", func(t *testing.T) {
				t.Parallel()
				p := NewHuggingFaceProvider(key, model, "")
				p.SetMeter(newCostTracker(t))
				resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
				if err != nil {
					t.Fatalf("chat: %v", err)
				}
				if resp.Content == "" {
					t.Fatal("empty response")
				}
				t.Logf("response: %s", resp.Content)
			})

			t.Run("StructuredOutput", func(t *testing.T) {
				t.Parallel()
				p := NewHuggingFaceProvider(key, model, "")
				p.SetMeter(newCostTracker(t))
				var out greeting
				err := p.CreateStructuredOutput(ctx, "greet me briefly", "respond with a greeting", &out)
				if err != nil {
					t.Fatalf("structured output: %v", err)
				}
				if out.Message == "" {
					t.Fatal("empty message")
				}
				t.Logf("greeting: %s", out.Message)
			})

			t.Run("StructuredOutputFromSchema", func(t *testing.T) {
				t.Parallel()
				p := NewHuggingFaceProvider(key, model, "")
				p.SetMeter(newCostTracker(t))
				result, err := p.CreateStructuredOutputFromSchema(ctx, "greet me briefly", "respond with a greeting", testSchema)
				if err != nil {
					t.Fatalf("structured output from schema: %v", err)
				}
				if result["message"] == nil || result["message"] == "" {
					t.Fatal("empty message")
				}
				t.Logf("greeting: %v", result["message"])
			})

			t.Run("ChatWithTools", func(t *testing.T) {
				t.Parallel()
				p := NewHuggingFaceProvider(key, model, "")
				p.SetMeter(newCostTracker(t))
				tools := []Tool{{
					Name:        "get_weather",
					Description: "Get weather for a city",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
						"required": []string{"city"},
					},
				}}
				resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "what's the weather in tokyo?"}}, tools)
				if err != nil {
					t.Fatalf("chat with tools: %v", err)
				}
				if resp.Content == "" && len(resp.ToolCalls) == 0 {
					t.Fatal("no content or tool calls")
				}
				if len(resp.ToolCalls) > 0 {
					t.Logf("tool call: %s(%s)", resp.ToolCalls[0].Name, string(resp.ToolCalls[0].Arguments))
				}
			})

			t.Run("Stream", func(t *testing.T) {
				t.Parallel()
				p := NewHuggingFaceProvider(key, model, "")
				p.SetMeter(newCostTracker(t))
				chunks, result := Stream(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
				var got strings.Builder
				for chunk, err := range chunks {
					if err != nil {
						t.Fatalf("stream chunk error: %v", err)
					}
					got.WriteString(chunk.Text)
				}
				if got.Len() == 0 {
					t.Fatal("no streamed text")
				}
				t.Logf("streamed: %s", got.String())
				resp, err := result.Response()
				if err != nil {
					t.Fatalf("stream response: %v", err)
				}
				if resp.Content != got.String() {
					t.Errorf("response %q != streamed %q", resp.Content, got.String())
				}
			})
		})
	}
}

// --- Ollama --------------------------------------------------------------

type ollamaTestModel struct {
	name           string
	structured     bool
	stream         bool
}

func TestOllama_Integration(t *testing.T) {
	baseURL := os.Getenv("OLLAMA_BASE_URL")
	if baseURL == "" {
		t.Skip("OLLAMA_BASE_URL not set")
	}
	t.Parallel()
	ctx := context.Background()
	models := []ollamaTestModel{
		{name: "qwen3.5:0.8b", structured: false, stream: true},
		{name: "gpt-oss:20b", structured: true, stream: true},
		{name: "gemma4:e4b", structured: false, stream: true},
	}

	for _, m := range models {
		m := m
		t.Run(m.name, func(t *testing.T) {
			t.Parallel()

			t.Run("Chat", func(t *testing.T) {
				t.Parallel()
				p := NewOllamaProvider(baseURL, m.name)
				p.SetMeter(newCostTracker(t))
				resp, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
				if err != nil {
					t.Fatalf("chat: %v", err)
				}
				if resp.Content == "" {
					t.Fatal("empty response")
				}
				t.Logf("response: %s", resp.Content)
			})

			if m.structured {
				t.Run("StructuredOutput", func(t *testing.T) {
					t.Parallel()
					p := NewOllamaProvider(baseURL, m.name)
					p.SetMeter(newCostTracker(t))
					var out greeting
					err := p.CreateStructuredOutput(ctx, "greet me briefly", "respond with a greeting", &out)
					if err != nil {
						t.Fatalf("structured output: %v", err)
					}
					if out.Message == "" {
						t.Fatal("empty message")
					}
					t.Logf("greeting: %s", out.Message)
				})
			}

			if m.stream {
				t.Run("Stream", func(t *testing.T) {
					t.Parallel()
					p := NewOllamaProvider(baseURL, m.name)
					p.SetMeter(newCostTracker(t))
					chunks, result := Stream(ctx, p, []Message{{Role: RoleUser, Content: "say hi in one word"}}, nil)
					var got strings.Builder
					for chunk, err := range chunks {
						if err != nil {
							t.Fatalf("stream chunk error: %v", err)
						}
						got.WriteString(chunk.Text)
					}
					if got.Len() == 0 {
						t.Fatal("no streamed text")
					}
					t.Logf("streamed: %s", got.String())
					resp, err := result.Response()
					if err != nil {
						t.Fatalf("stream response: %v", err)
					}
					if resp.Content != got.String() {
						t.Errorf("response %q != streamed %q", resp.Content, got.String())
					}
				})
			}
		})
	}
}

// --- fal ------------------------------------------------------------------

func TestFal_Integration(t *testing.T) {
	key := os.Getenv("FAL_API_KEY")
	if key == "" {
		t.Skip("FAL_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	// ~$0.24 per run (6s @ 1080p on LTX-2.3 fast).
	t.Run("VideoGenerate", func(t *testing.T) {
		t.Parallel()
		p := newFalVideoProvider(key, "")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:         "a solid blue square slowly rotating on a black background",
			Image:          testPNGBytes(t),
			ImageMediaType: "image/png",
			Duration:       6,
			Resolution:     "1080p",
			AspectRatio:    "16:9",
		})
		if err != nil {
			t.Fatalf("video generate: %v", err)
		}
		if res.URL == "" || !strings.HasPrefix(res.URL, "http") {
			t.Fatalf("expected a hosted video url, got %q", res.URL)
		}
		if res.Duration <= 0 || res.CostUSD <= 0 {
			t.Fatalf("expected duration + cost, got %+v", res)
		}
		t.Logf("video: %s (%dx%d, %.1fs, $%.3f, %s)", res.URL, res.Width, res.Height, res.Duration, res.CostUSD, res.Elapsed)
	})

	// ~$0.24 per run (6s @ 1080p on LTX-2.3 fast, text-only).
	t.Run("TextToVideo", func(t *testing.T) {
		t.Parallel()
		p := newFalVideoProvider(key, "fal-ai/ltx-2.3/text-to-video/fast")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:      "a solid blue square slowly rotating on a black background",
			Duration:    6,
			Resolution:  "1080p",
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("text-to-video: %v", err)
		}
		if res.URL == "" || !strings.HasPrefix(res.URL, "http") {
			t.Fatalf("expected a hosted video url, got %q", res.URL)
		}
		if res.Duration <= 0 || res.CostUSD <= 0 {
			t.Fatalf("expected duration + cost, got %+v", res)
		}
		t.Logf("video: %s (%dx%d, %.1fs, $%.3f, %s)", res.URL, res.Width, res.Height, res.Duration, res.CostUSD, res.Elapsed)
	})

	// ~$0.40 per run (5s @ 768p on minimax h3-max).
	t.Run("VideoGenerate_MinimaxH3Max", func(t *testing.T) {
		t.Parallel()
		p := newFalVideoProvider(key, "minimax/h3-max/image-to-video")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:         "a solid blue square slowly rotating on a black background",
			Image:          testPNGBytesSize(t, 512, 512),
			ImageMediaType: "image/png",
			Duration:       5,
			Resolution:     "768P",
		})
		if err != nil {
			t.Fatalf("video generate: %v", err)
		}
		if res.URL == "" || !strings.HasPrefix(res.URL, "http") {
			t.Fatalf("expected a hosted video url, got %q", res.URL)
		}
		if res.Duration <= 0 || res.CostUSD <= 0 {
			t.Fatalf("expected duration + cost, got %+v", res)
		}
		t.Logf("video: %s (%dx%d, %.1fs, $%.3f, %s)", res.URL, res.Width, res.Height, res.Duration, res.CostUSD, res.Elapsed)
	})
}

// --- gemini video -----------------------------------------------------------

func TestGeminiVideo_Integration(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("VideoGenerate", func(t *testing.T) {
		t.Parallel()
		p := newGeminiVideoProvider(key, "")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:      "a solid blue square slowly rotating on a black background",
			Image:       testPNGBytes(t),
			Duration:    5,
			Resolution:  "720p",
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("video generate: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		if res.CostUSD <= 0 {
			t.Fatalf("expected cost, got %+v", res)
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})

	t.Run("TextToVideo", func(t *testing.T) {
		t.Parallel()
		p := newGeminiVideoProvider(key, "")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:      "a solid blue square slowly rotating on a black background",
			Duration:    5,
			Resolution:  "720p",
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("text-to-video: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		if res.CostUSD <= 0 {
			t.Fatalf("expected cost, got %+v", res)
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})
}

func TestVideoRouter_Integration(t *testing.T) {
	falKey := os.Getenv("FAL_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if falKey == "" && geminiKey == "" {
		t.Skip("FAL_API_KEY and GEMINI_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	var providers []VideoRouterOption
	if falKey != "" {
		fal, err := NewVideoProvider("fal", falKey, "")
		if err != nil {
			t.Fatalf("fal provider: %v", err)
		}
		providers = append(providers, WithRoute(fal))
	}
	if geminiKey != "" {
		gemini, err := NewVideoProvider("gemini", geminiKey, "")
		if err != nil {
			t.Fatalf("gemini provider: %v", err)
		}
		providers = append(providers, WithRoute(gemini))
	}

	t.Run("RouteByPrice", func(t *testing.T) {
		t.Parallel()
		opts := append(providers, RouteByPrice(1.0))
		router, err := NewVideoRouter(opts...)
		if err != nil {
			t.Fatal(err)
		}
		router.SetMeter(newCostTracker(t))

		res, err := router.Generate(ctx, VideoRequest{
			Prompt:     "a solid blue square slowly rotating on a black background",
			Duration:   5,
			Resolution: "720p",
		})
		if err != nil {
			t.Fatalf("route by price: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		t.Logf("routed to %s: %s (%.1fs, $%.3f, %s)", res.Model, res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})

	t.Run("RouteByAvailability", func(t *testing.T) {
		t.Parallel()
		opts := append(providers, RouteByAvailability(1.0))
		router, err := NewVideoRouter(opts...)
		if err != nil {
			t.Fatal(err)
		}
		router.SetMeter(newCostTracker(t))

		res, err := router.Generate(ctx, VideoRequest{
			Prompt:     "a solid blue square slowly rotating on a black background",
			Duration:   5,
			Resolution: "720p",
		})
		if err != nil {
			t.Fatalf("route by availability: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		t.Logf("routed to %s: %s (%.1fs, $%.3f, %s)", res.Model, res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})
}

func TestMinimaxVideo_Integration(t *testing.T) {
	key := os.Getenv("MINIMAX_API_KEY")
	if key == "" {
		t.Skip("MINIMAX_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("TextToVideo", func(t *testing.T) {
		t.Parallel()
		p := newMinimaxVideoProvider(key, "")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:      "a solid blue square slowly rotating on a black background",
			Duration:    5,
			Resolution:  "768P",
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("text-to-video: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		if res.CostUSD <= 0 {
			t.Fatalf("expected cost, got %+v", res)
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})

	t.Run("ImageToVideo", func(t *testing.T) {
		t.Parallel()
		p := newMinimaxVideoProvider(key, "")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:     "the blue square begins spinning faster",
			Image:      testPNGBytes(t),
			Duration:   5,
			Resolution: "768P",
		})
		if err != nil {
			t.Fatalf("image-to-video: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})
}

func TestVeoVideo_Integration(t *testing.T) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	t.Parallel()
	ctx := context.Background()

	t.Run("TextToVideo", func(t *testing.T) {
		t.Parallel()
		p := newVeoVideoProvider(key, "veo-3.1-fast-generate-preview")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:      "a solid blue square slowly rotating on a black background",
			Duration:    4,
			Resolution:  "720p",
			AspectRatio: "16:9",
		})
		if err != nil {
			t.Fatalf("text-to-video: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		if res.CostUSD <= 0 {
			t.Fatalf("expected cost, got %+v", res)
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})

	t.Run("ImageToVideo", func(t *testing.T) {
		t.Parallel()
		p := newVeoVideoProvider(key, "veo-3.1-fast-generate-preview")
		p.SetMeter(newCostTracker(t))
		res, err := p.Generate(ctx, VideoRequest{
			Prompt:     "the blue square begins spinning faster",
			Image:      testPNGBytes(t),
			Duration:   4,
			Resolution: "720p",
		})
		if err != nil {
			t.Fatalf("image-to-video: %v", err)
		}
		if res.URL == "" {
			t.Fatal("expected a video url")
		}
		t.Logf("video: %s (%.1fs, $%.3f, %s)", res.URL, res.Duration, res.CostUSD, res.Elapsed)
	})
}
