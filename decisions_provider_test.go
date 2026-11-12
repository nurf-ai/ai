package ai

import (
	"encoding/json"
	"testing"
)

// Zeros are answers: a noul of 0, a score at the lowest level and a zero
// confidence survive a marshal, and each type writes only its own fields.
func TestDecisionsAnswerMarshalKeepsZeros(t *testing.T) {
	cases := []struct {
		name string
		in   DecisionsAnswer
		want string
	}{
		{"noul of 0", DecisionsAnswer{Type: QuestionNoul}, `{"type":"noul","noul":0}`},
		{"noul with confidence", DecisionsAnswer{Type: QuestionNoul, Noul: 0.94, Confidence: 0.94}, `{"type":"noul","noul":0.94,"confidence":0.94}`},
		{"noul", DecisionsAnswer{Type: QuestionNoul, Noul: 0.82}, `{"type":"noul","noul":0.82}`},
		{"choice with zero confidence",
			DecisionsAnswer{Type: QuestionChoice, Choice: "top", Probabilities: map[string]float64{"top": 1, "shoes": 0}},
			`{"type":"choice","choice":"top","probabilities":{"shoes":0,"top":1},"confidence":0}`},
		{"score at the lowest level",
			DecisionsAnswer{Type: QuestionScore, Legend: map[string]string{"0": "calm", "1": "angry"}, Probabilities: map[string]float64{"0": 1, "1": 0}, Confidence: 0.9},
			`{"type":"score","score":0,"legend":{"0":"calm","1":"angry"},"probabilities":{"0":1,"1":0},"confidence":0.9}`},
		{"unknown type", DecisionsAnswer{Type: "rank", Choice: "a"}, `{"type":"rank","choice":"a"}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := json.Marshal(c.in)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != c.want {
				t.Fatalf("got %s, want %s", got, c.want)
			}
		})
	}

	// the provider's decode and the host's re-encode keep the number
	var res DecisionsResult
	if err := json.Unmarshal([]byte(`{"model":"jev","answers":{"x":{"type":"noul","noul":0}}}`), &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got, err := json.Marshal(res.Answers)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if want := `{"x":{"type":"noul","noul":0}}`; string(got) != want {
		t.Fatalf("round trip: got %s, want %s", got, want)
	}
}
