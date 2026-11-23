package ai

import (
	"math"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestEstimateOpenAIImageCost(t *testing.T) {
	// gpt-image-2: text in $5/M, image in $8/M, image out $30/M.
	u := openai.ImageResponseUsage{
		InputTokens:        1100,
		OutputTokens:       200,
		InputTokensDetails: openai.ImageResponseInputTokensDetails{TextTokens: 100, ImageTokens: 1000},
	}
	want := (100*5.0 + 1000*8.0 + 200*30.0) / 1_000_000
	if got := estimateOpenAIImageCost("gpt-image-2", u); math.Abs(got-want) > 1e-9 {
		t.Errorf("by tokens = %f, want %f", got, want)
	}
	if got := estimateOpenAIImageCost("gpt-image-2", openai.ImageResponseUsage{}); got != 0.006 {
		t.Errorf("no usage = %f, want 0.006 (flat)", got)
	}
	if got := estimateOpenAIImageCost("unknown-model", u); got != 0 {
		t.Errorf("unknown model = %f, want 0", got)
	}
}
