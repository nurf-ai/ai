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

// DecisionsQuestion is one typed question in a Decisions request.
type DecisionsQuestion struct {
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
func Noul(instructions any) DecisionsQuestion {
	return DecisionsQuestion{Type: QuestionNoul, Instructions: instructions}
}

// NoulWith builds a noul question with explicit true/false descriptions.
func NoulWith(instructions any, trueDesc, falseDesc string) DecisionsQuestion {
	return DecisionsQuestion{
		Type:         QuestionNoul,
		Instructions: instructions,
		Criteria:     NoulCriteria{True: trueDesc, False: falseDesc},
	}
}

// Choice builds a single-selection question. Each key is an option; a nil
// value means no description for that option.
func Choice(instructions any, options map[string]*string) DecisionsQuestion {
	return DecisionsQuestion{
		Type:         QuestionChoice,
		Instructions: instructions,
		Criteria:     options,
	}
}

// Score builds an ordinal rating question. Levels are ordered low→high,
// minimum 2.
func Score(instructions any, levels []string) DecisionsQuestion {
	return DecisionsQuestion{
		Type:         QuestionScore,
		Instructions: instructions,
		Criteria:     levels,
	}
}

// DecisionsAnswer is the typed response for one question. Fields are populated
// based on Type: noul → Noul/Confidence; choice → Choice/Probabilities/Confidence;
// score → Score/Legend/Probabilities/Confidence.
type DecisionsAnswer struct {
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
func (a DecisionsAnswer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case QuestionNoul:
		return json.Marshal(struct {
			Type       string  `json:"type"`
			Noul       float64 `json:"noul"`
			Confidence float64 `json:"confidence,omitempty"`
		}{a.Type, a.Noul, a.Confidence})
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
	type plain DecisionsAnswer // same fields, no MarshalJSON: no recursion
	return json.Marshal(plain(a))
}

// DecisionsContent is a Gemini Decisions input block. Data holds base64-encoded
// bytes; URI accepts a Files API URI or an HTTPS URL. Set MIMEType for media.
type DecisionsContent struct {
	Type       string `json:"type"`
	Text       string `json:"text,omitempty"`
	Data       string `json:"data,omitempty"`
	URI        string `json:"uri,omitempty"`
	MIMEType   string `json:"mime_type,omitempty"`
	Resolution string `json:"resolution,omitempty"`
}

// DecisionsRequest is the input to a Decisions call.
type DecisionsRequest struct {
	// State is shared context. Gemini accepts text or DecisionsContent blocks;
	// other values are serialized as JSON text. Typesafe receives State as JSON.
	State     any                          `json:"state"`
	Questions map[string]DecisionsQuestion `json:"questions"`
}

// DecisionsUsage tracks token consumption for a Decisions call.
type DecisionsUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// DecisionsResult is the output of a Decisions call.
type DecisionsResult struct {
	Model   string                     `json:"model"`
	Answers map[string]DecisionsAnswer `json:"answers"`
	Usage   DecisionsUsage             `json:"usage"`
}

// DecisionsProvider evaluates content against typed questions and returns
// structured answers with calibrated confidence.
type DecisionsProvider interface {
	Judge(ctx context.Context, req *DecisionsRequest) (*DecisionsResult, error)
}

// DecisionsMeterable is implemented by Decisions providers that accept a meter hook.
type DecisionsMeterable interface {
	SetMeter(MeterHook)
}

// SetDecisionsMeter attaches a meter hook to any DecisionsProvider that
// satisfies DecisionsMeterable.
func SetDecisionsMeter(j DecisionsProvider, hook MeterHook) {
	if hook == nil || j == nil {
		return
	}
	if m, ok := j.(DecisionsMeterable); ok {
		m.SetMeter(hook)
	}
}

// NewDecisionsProvider creates a DecisionsProvider from a provider name.
func NewDecisionsProvider(providerName, apiKey, model string) DecisionsProvider {
	switch providerName {
	case "gemini":
		var opts []GeminiDecisionsOption
		if model != "" {
			opts = append(opts, WithGeminiDecisionsModel(model))
		}
		return NewGeminiDecisionsProvider(apiKey, opts...)
	case "typesafe":
		var opts []TypesafeOption
		if model != "" {
			opts = append(opts, WithTypesafeModel(model))
		}
		return NewTypesafeDecisionsProvider(apiKey, opts...)
	default:
		return nil
	}
}
