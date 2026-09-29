package ai

import (
	"context"
	"encoding/json"
)

const (
	QuestionNoul   = "noul"
	QuestionChoice = "choice"
	QuestionScore  = "score"
)

// SystemOneQuestion is one typed question in a System One request.
type SystemOneQuestion struct {
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
func Noul(instructions any) SystemOneQuestion {
	return SystemOneQuestion{Type: QuestionNoul, Instructions: instructions}
}

// NoulWith builds a noul question with explicit true/false descriptions.
func NoulWith(instructions any, trueDesc, falseDesc string) SystemOneQuestion {
	return SystemOneQuestion{
		Type:         QuestionNoul,
		Instructions: instructions,
		Criteria:     NoulCriteria{True: trueDesc, False: falseDesc},
	}
}

// Choice builds a single-selection question. Each key is an option; a nil
// value means no description for that option.
func Choice(instructions any, options map[string]*string) SystemOneQuestion {
	return SystemOneQuestion{
		Type:         QuestionChoice,
		Instructions: instructions,
		Criteria:     options,
	}
}

// Score builds an ordinal rating question. Levels are ordered low→high,
// minimum 2.
func Score(instructions any, levels []string) SystemOneQuestion {
	return SystemOneQuestion{
		Type:         QuestionScore,
		Instructions: instructions,
		Criteria:     levels,
	}
}

// SystemOneAnswer is the typed response for one question. Fields are populated
// based on Type: noul → Noul; choice → Choice/Probabilities/Confidence;
// score → Score/Legend/Probabilities/Confidence.
type SystemOneAnswer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
}

// MarshalJSON writes the fields of the answer's type, its numbers always: a
// noul of 0, a score at the lowest level and a confidence of 0 are answers,
// and the struct's omitempty would drop them. An unknown type writes every
// field as tagged.
func (a SystemOneAnswer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case QuestionNoul:
		return json.Marshal(struct {
			Type string  `json:"type"`
			Noul float64 `json:"noul"`
		}{a.Type, a.Noul})
	case QuestionChoice:
		return json.Marshal(struct {
			Type          string             `json:"type"`
			Choice        string             `json:"choice,omitempty"`
			Probabilities map[string]float64 `json:"probabilities,omitempty"`
			Confidence    float64            `json:"confidence"`
		}{a.Type, a.Choice, a.Probabilities, a.Confidence})
	case QuestionScore:
		return json.Marshal(struct {
			Type          string             `json:"type"`
			Score         float64            `json:"score"`
			Legend        map[string]string  `json:"legend,omitempty"`
			Probabilities map[string]float64 `json:"probabilities,omitempty"`
			Confidence    float64            `json:"confidence"`
		}{a.Type, a.Score, a.Legend, a.Probabilities, a.Confidence})
	}
	type plain SystemOneAnswer // same fields, no MarshalJSON: no recursion
	return json.Marshal(plain(a))
}

// SystemOneRequest is the input to a System One call.
type SystemOneRequest struct {
	State     any                          `json:"state"`
	Questions map[string]SystemOneQuestion `json:"questions"`
}

// SystemOneUsage tracks token consumption for a System One call.
type SystemOneUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// SystemOneResult is the output of a System One call.
type SystemOneResult struct {
	Model   string                     `json:"model"`
	Answers map[string]SystemOneAnswer `json:"answers"`
	Usage   SystemOneUsage             `json:"usage"`
}

// SystemOneProvider evaluates content against typed questions and returns
// structured answers with calibrated confidence.
type SystemOneProvider interface {
	Judge(ctx context.Context, req *SystemOneRequest) (*SystemOneResult, error)
}

// SystemOneMeterable is implemented by System One providers that accept a meter hook.
type SystemOneMeterable interface {
	SetMeter(MeterHook)
}

// SetSystemOneMeter attaches a meter hook to any SystemOneProvider that
// satisfies SystemOneMeterable.
func SetSystemOneMeter(j SystemOneProvider, hook MeterHook) {
	if hook == nil || j == nil {
		return
	}
	if m, ok := j.(SystemOneMeterable); ok {
		m.SetMeter(hook)
	}
}

// NewSystemOneProvider creates a SystemOneProvider from a provider name.
func NewSystemOneProvider(providerName, apiKey, model string) SystemOneProvider {
	switch providerName {
	case "typesafe":
		var opts []TypesafeOption
		if model != "" {
			opts = append(opts, WithTypesafeModel(model))
		}
		return NewTypesafeSystemOneProvider(apiKey, opts...)
	default:
		return nil
	}
}
