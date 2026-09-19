package ai

import "context"

const (
	QuestionNoul   = "noul"
	QuestionChoice = "choice"
	QuestionScore  = "score"
)

// JudgmentQuestion is one typed question in a judgment request.
type JudgmentQuestion struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// NoulCriteria describes what true/false mean for a noul question.
type NoulCriteria struct {
	True  string `json:"true"`
	False string `json:"false"`
}

// Noul builds a boolean-probability question.
func Noul(instructions any) JudgmentQuestion {
	return JudgmentQuestion{Type: QuestionNoul, Instructions: instructions}
}

// NoulWith builds a noul question with explicit true/false descriptions.
func NoulWith(instructions any, trueDesc, falseDesc string) JudgmentQuestion {
	return JudgmentQuestion{
		Type:         QuestionNoul,
		Instructions: instructions,
		Criteria:     NoulCriteria{True: trueDesc, False: falseDesc},
	}
}

// Choice builds a single-selection question. Each key is an option; a nil
// value means no description for that option.
func Choice(instructions any, options map[string]*string) JudgmentQuestion {
	return JudgmentQuestion{
		Type:         QuestionChoice,
		Instructions: instructions,
		Criteria:     options,
	}
}

// Score builds an ordinal rating question. Levels are ordered low→high,
// minimum 2.
func Score(instructions any, levels []string) JudgmentQuestion {
	return JudgmentQuestion{
		Type:         QuestionScore,
		Instructions: instructions,
		Criteria:     levels,
	}
}

// JudgmentAnswer is the typed response for one question. Fields are populated
// based on Type: noul → Noul; choice → Choice/Probabilities/Confidence;
// score → Score/Legend/Probabilities/Confidence.
type JudgmentAnswer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
}

// JudgmentRequest is the input to a judgment call.
type JudgmentRequest struct {
	State     any                         `json:"state"`
	Questions map[string]JudgmentQuestion `json:"questions"`
}

// JudgmentUsage tracks token consumption for a judgment call.
type JudgmentUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// JudgmentResult is the output of a judgment call.
type JudgmentResult struct {
	Model   string                      `json:"model"`
	Answers map[string]JudgmentAnswer   `json:"answers"`
	Usage   JudgmentUsage               `json:"usage"`
}

// JudgmentProvider evaluates content against typed questions and returns
// structured answers with calibrated confidence.
type JudgmentProvider interface {
	Judge(ctx context.Context, req *JudgmentRequest) (*JudgmentResult, error)
}

// JudgmentMeterable is implemented by judgment providers that accept a meter hook.
type JudgmentMeterable interface {
	SetMeter(MeterHook)
}

// SetJudgmentMeter attaches a meter hook to any JudgmentProvider that
// satisfies JudgmentMeterable.
func SetJudgmentMeter(j JudgmentProvider, hook MeterHook) {
	if hook == nil || j == nil {
		return
	}
	if m, ok := j.(JudgmentMeterable); ok {
		m.SetMeter(hook)
	}
}

// NewJudgmentProvider creates a JudgmentProvider from a provider name.
func NewJudgmentProvider(providerName, apiKey string) JudgmentProvider {
	switch providerName {
	case "typesafe":
		return NewTypesafeJudgmentProvider(apiKey)
	default:
		return nil
	}
}
