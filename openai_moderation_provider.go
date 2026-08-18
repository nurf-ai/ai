package ai

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

type OpenAIModerationProvider struct {
	raw *openai.Client
}

func NewOpenAIModerationProvider(apiKey string) *OpenAIModerationProvider {
	return &OpenAIModerationProvider{raw: openai.NewClient(apiKey)}
}

func (p *OpenAIModerationProvider) Check(ctx context.Context, input string) (*ModerationResult, error) {
	resp, err := p.raw.Moderations(ctx, openai.ModerationRequest{
		Input: input,
		Model: openai.ModerationOmniLatest,
	})
	if err != nil {
		return nil, fmt.Errorf("openai moderation: %w", err)
	}
	if len(resp.Results) == 0 {
		return &ModerationResult{}, nil
	}

	r := resp.Results[0]
	cats := map[string]bool{}
	if r.Categories.Hate {
		cats["hate"] = true
	}
	if r.Categories.HateThreatening {
		cats["hate/threatening"] = true
	}
	if r.Categories.Harassment {
		cats["harassment"] = true
	}
	if r.Categories.HarassmentThreatening {
		cats["harassment/threatening"] = true
	}
	if r.Categories.SelfHarm {
		cats["self-harm"] = true
	}
	if r.Categories.SelfHarmIntent {
		cats["self-harm/intent"] = true
	}
	if r.Categories.SelfHarmInstructions {
		cats["self-harm/instructions"] = true
	}
	if r.Categories.Sexual {
		cats["sexual"] = true
	}
	if r.Categories.SexualMinors {
		cats["sexual/minors"] = true
	}
	if r.Categories.Violence {
		cats["violence"] = true
	}
	if r.Categories.ViolenceGraphic {
		cats["violence/graphic"] = true
	}

	if r.Flagged {
		logger.Warn("flagged", zap.Any("categories", cats))
	}

	return &ModerationResult{
		Flagged:    r.Flagged,
		Categories: cats,
	}, nil
}
