package ai

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// TestPricingCoverage verifies that every non-image model in models.json
// declares max_input_tokens. Catches entries where the field was omitted.
func TestPricingCoverage(t *testing.T) {
	for model, p := range pricingTable {
		if p.FlatPerImage > 0 || p.PerVideoSecond > 0 || stripDateSuffix(model) != model {
			continue
		}
		if p.MaxInputTokens == 0 {
			t.Errorf("model %q has pricing but no max_input_tokens in models.json", model)
		}
	}
}

func TestEstimateCostFull_BasicMath(t *testing.T) {
	tests := []struct {
		name   string
		model  string
		in     int
		out    int
		expect float64
	}{
		// Anthropic, no cache
		{"haiku 1M in", "claude-haiku-4-5-20251001", 1_000_000, 0, 1.00},
		{"haiku 1M out", "claude-haiku-4-5-20251001", 0, 1_000_000, 5.00},
		{"haiku 500k+500k", "claude-haiku-4-5-20251001", 500_000, 500_000, 3.00},
		{"sonnet 1M+1M", "claude-sonnet-4-5-20250514", 1_000_000, 1_000_000, 18.00},
		// OpenAI, no cache
		{"4o-mini 1M+1M", "gpt-4o-mini", 1_000_000, 1_000_000, 0.75},
		{"4o 100k+100k", "gpt-4o", 100_000, 100_000, 1.25},
		{"haiku 0+0", "claude-haiku-4-5-20251001", 0, 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EstimateCostFull(tt.model, tt.in, tt.out, 0, 0)
			if math.Abs(got-tt.expect) > 0.001 {
				t.Errorf("got %f, want %f", got, tt.expect)
			}
		})
	}
}

func TestEstimateCostFull_CacheCreation(t *testing.T) {
	// Anthropic charges 1.25× input on cache write.
	// claude-haiku-4-5 has Input=$1.00/M → CacheCreation=$1.25/M.
	got := EstimateCostFull("claude-haiku-4-5", 0, 0, 1_000_000, 0)
	if math.Abs(got-1.25) > 0.001 {
		t.Errorf("haiku 1M cw = %f, want 1.25", got)
	}
}

func TestEstimateCostFull_CacheRead(t *testing.T) {
	// Anthropic charges 0.10× input on cache read.
	// claude-haiku-4-5 has Input=$1.00/M → CacheRead=$0.10/M.
	got := EstimateCostFull("claude-haiku-4-5", 0, 0, 0, 1_000_000)
	if math.Abs(got-0.10) > 0.001 {
		t.Errorf("haiku 1M cr = %f, want 0.10", got)
	}
}

func TestEstimateCostFull_Blend(t *testing.T) {
	// Realistic opus-4-6 turn: 5k fresh input, 1k output, 30k cache-read hit.
	// opus-4-6 prices: in=$5/M, out=$25/M, cw=$6.25/M, cr=$0.50/M.
	// expect = (5000*5 + 1000*25 + 0*6.25 + 30000*0.50) / 1_000_000
	//       = (25000 + 25000 + 0 + 15000) / 1_000_000
	//       = 65000 / 1_000_000 = 0.065
	got := EstimateCostFull("claude-opus-4-6", 5000, 1000, 0, 30000)
	want := 0.065
	if math.Abs(got-want) > 0.0001 {
		t.Errorf("opus-4-6 realistic turn = %f, want %f", got, want)
	}
}

func TestEstimateCostFull_FlatImage(t *testing.T) {
	tests := []struct {
		model  string
		expect float64
	}{
		{"gpt-image-1", 0.04},
		{"gemini-2.5-flash-image", 0.039},
	}
	for _, tt := range tests {
		got := EstimateCostFull(tt.model, 999, 999, 999, 999)
		if got != tt.expect {
			t.Errorf("%q = %f, want %f", tt.model, got, tt.expect)
		}
	}
}

func TestEstimateVideoCostByTokens(t *testing.T) {
	// gemini-omni-1.1-flash: input=$1.50/M, video_output=$17.50/M
	// 11 input + 57920 video output → 11*1.50/1M + 57920*17.50/1M
	got := EstimateVideoCostByTokens("gemini-omni-1.1-flash", 11, 57920)
	want := float64(11)*1.50/1_000_000 + float64(57920)*17.50/1_000_000
	if math.Abs(got-want) > 0.0001 {
		t.Errorf("gemini video by tokens = %f, want %f", got, want)
	}
	// unknown model returns 0
	if got := EstimateVideoCostByTokens("unknown", 100, 100); got != 0 {
		t.Errorf("unknown model = %f, want 0", got)
	}
}

func TestEstimateImageCostByTokens(t *testing.T) {
	// gpt-image-1: image_output=$40/M → 1000 tokens = $0.04
	got := EstimateImageCostByTokens("gpt-image-1", 1000)
	want := float64(1000) * 40.0 / 1_000_000
	if math.Abs(got-want) > 0.0001 {
		t.Errorf("gpt-image-1 by tokens = %f, want %f", got, want)
	}
	// with 0 tokens, falls back to flat_per_image
	got = EstimateImageCostByTokens("gpt-image-1", 0)
	if got != 0.04 {
		t.Errorf("gpt-image-1 no tokens = %f, want 0.04 (flat)", got)
	}
	// gemini image model
	got = EstimateImageCostByTokens("gemini-2.5-flash-image", 1300)
	want = float64(1300) * 30.0 / 1_000_000
	if math.Abs(got-want) > 0.0001 {
		t.Errorf("gemini image by tokens = %f, want %f", got, want)
	}
}

func TestEstimateCostFull_UnknownModel(t *testing.T) {
	var buf bytes.Buffer
	core := zapcore.NewCore(
		zapcore.NewJSONEncoder(zap.NewProductionEncoderConfig()),
		zapcore.AddSync(&buf),
		zapcore.WarnLevel,
	)
	SetLogger(zap.New(core))
	defer SetLogger(zap.NewNop())

	got := EstimateCostFull("unknown-model-xyz", 1000, 1000, 0, 0)
	if got != 0 {
		t.Errorf("unknown model cost = %f, want 0", got)
	}
	if !strings.Contains(buf.String(), "unknown model") {
		t.Errorf("expected unknown-model warn log; got: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "unknown-model-xyz") {
		t.Errorf("expected model name in warn log; got: %q", buf.String())
	}
}

// TestEstimateCostFull_OpenAICacheRead checks that GPT-5's published 10×
// discount on cached tokens is wired through correctly.
// gpt-5: Input=$1.25/M, CacheRead=$0.125/M.
// 1M cached read = $0.125.
func TestEstimateCostFull_OpenAICacheRead(t *testing.T) {
	got := EstimateCostFull("gpt-5", 0, 0, 0, 1_000_000)
	if math.Abs(got-0.125) > 0.001 {
		t.Errorf("gpt-5 1M cr = %f, want 0.125", got)
	}
}

func TestMaxInputTokensLLM(t *testing.T) {
	tests := []struct {
		company string
		model   string
		want    int64
	}{
		{"anthropic", "claude-haiku-4-5", 200_000},
		{"anthropic", "claude-sonnet-5", 1_000_000},
		{"openai", "gpt-4o-mini", 128_000},
		{"gemini", "gemini-3.6-flash", 1_048_576},
		{"huggingface", "moonshotai/Kimi-K3", 1_048_576},
	}
	for _, tt := range tests {
		t.Run(tt.company+"/"+tt.model, func(t *testing.T) {
			got, err := MaxInputTokensLLM(tt.company, tt.model)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func TestMaxInputTokensLLM_DateSuffix(t *testing.T) {
	got, err := MaxInputTokensLLM("anthropic", "claude-haiku-4-5-20251001")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 200_000 {
		t.Errorf("got %d, want 200000", got)
	}
}

func TestMaxInputTokensLLM_Unknown(t *testing.T) {
	_, err := MaxInputTokensLLM("openai", "nonexistent-model")
	if err == nil {
		t.Fatal("expected error for unknown model")
	}
}

func TestModelsForProvider(t *testing.T) {
	models := ModelsForProvider("anthropic")
	if len(models) == 0 {
		t.Fatal("expected at least one model for anthropic")
	}
	for _, m := range models {
		if strings.Contains(m, "/") {
			t.Errorf("model %q should not contain company prefix", m)
		}
		if stripDateSuffix(m) != m {
			t.Errorf("dated model %q should be filtered out", m)
		}
	}

	empty := ModelsForProvider("nonexistent")
	if len(empty) != 0 {
		t.Errorf("expected no models for nonexistent provider, got %d", len(empty))
	}
}
