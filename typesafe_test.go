package ai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTypesafeJudge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("want POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/systemone" {
			t.Fatalf("want /v1/systemone, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("want Bearer test-key, got %q", got)
		}

		body, _ := io.ReadAll(r.Body)
		var req typesafeRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "jev-1.14.0" {
			t.Fatalf("want model jev-1.14.0, got %s", req.Model)
		}
		if _, ok := req.Questions["is_urgent"]; !ok {
			t.Fatal("missing question is_urgent")
		}
		if _, ok := req.Questions["category"]; !ok {
			t.Fatal("missing question category")
		}
		if _, ok := req.Questions["anger"]; !ok {
			t.Fatal("missing question anger")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(DecisionsResult{
			Model: "jev-1.14.0",
			Answers: map[string]DecisionsAnswer{
				"is_urgent": {Type: QuestionNoul, Noul: 0.92},
				"category": {
					Type:          QuestionChoice,
					Choice:        "billing",
					Probabilities: map[string]float64{"billing": 0.85, "technical": 0.10, "sales": 0.05},
					Confidence:    0.82,
				},
				"anger": {
					Type:          QuestionScore,
					Score:         1.6,
					Legend:        map[string]string{"0": "Calm", "1": "Frustrated", "2": "Very angry"},
					Probabilities: map[string]float64{"0": 0.05, "1": 0.3, "2": 0.65},
					Confidence:    0.78,
				},
			},
			Usage: DecisionsUsage{InputTokens: 100, OutputTokens: 50},
		})
	}))
	defer srv.Close()

	p := NewTypesafeDecisionsProvider("test-key",
		WithTypesafeBase(srv.URL),
	)

	billing := "related to billing"
	result, err := p.Judge(context.Background(), &DecisionsRequest{
		State: "Help! My payouts have been failing for 3 days.",
		Questions: map[string]DecisionsQuestion{
			"is_urgent": Noul("Does this convey urgency?"),
			"category":  Choice("What category?", map[string]*string{"billing": &billing, "technical": nil, "sales": nil}),
			"anger":     Score("How angry is the user?", []string{"Calm", "Frustrated", "Very angry"}),
		},
	})
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}

	if result.Model != "jev-1.14.0" {
		t.Errorf("model: want jev-1.14.0, got %s", result.Model)
	}

	// noul
	if a := result.Answers["is_urgent"]; a.Type != QuestionNoul || a.Noul != 0.92 {
		t.Errorf("is_urgent: got %+v", a)
	}
	// choice
	if a := result.Answers["category"]; a.Type != QuestionChoice || a.Choice != "billing" {
		t.Errorf("category: got %+v", a)
	}
	// score
	if a := result.Answers["anger"]; a.Type != QuestionScore || a.Score != 1.6 {
		t.Errorf("anger: got %+v", a)
	}
	// usage
	if result.Usage.InputTokens != 100 || result.Usage.OutputTokens != 50 {
		t.Errorf("usage: got %+v", result.Usage)
	}
}

func TestTypesafeJudgeMeter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(DecisionsResult{
			Model:   "jev-1.14.0",
			Answers: map[string]DecisionsAnswer{"q": {Type: QuestionNoul, Noul: 0.5}},
			Usage:   DecisionsUsage{InputTokens: 42, OutputTokens: 7},
		})
	}))
	defer srv.Close()

	var got UsageEvent
	p := NewTypesafeDecisionsProvider("k", WithTypesafeBase(srv.URL))
	p.SetMeter(func(ev UsageEvent) { got = ev })

	_, err := p.Judge(WithDebugSpanID(context.Background(), "span-1"), &DecisionsRequest{
		State:     "test",
		Questions: map[string]DecisionsQuestion{"q": Noul("yes?")},
	})
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if got.DebugSpanID != "span-1" {
		t.Errorf("debug_span_id: want span-1, got %q", got.DebugSpanID)
	}
	if got.Provider != "typesafe" {
		t.Errorf("provider: want typesafe, got %s", got.Provider)
	}
	if got.EstimatedCostUSD != float64(42)*0.042/1_000_000 {
		t.Errorf("cost = %.12f, want input-only price %.12f", got.EstimatedCostUSD, float64(42)*0.042/1_000_000)
	}
	if got.Model != "jev-1.14.0" {
		t.Errorf("model = %q", got.Model)
	}
	if got.TotalTokens != 49 {
		t.Errorf("total_tokens: want 49, got %d", got.TotalTokens)
	}
}

func TestTypesafeError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"invalid api key"}`))
	}))
	defer srv.Close()

	p := NewTypesafeDecisionsProvider("bad-key", WithTypesafeBase(srv.URL))
	_, err := p.Judge(context.Background(), &DecisionsRequest{
		State:     "x",
		Questions: map[string]DecisionsQuestion{"q": Noul("y")},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	kind, _ := ClassifyError(err)
	if kind != ErrAuth {
		t.Errorf("want ErrAuth, got %v", kind)
	}
}

func TestTypesafeNoulWithCriteria(t *testing.T) {
	q := NoulWith("Is this urgent?", "yes it's urgent", "not urgent")
	if q.Type != QuestionNoul {
		t.Errorf("type: want noul, got %s", q.Type)
	}
	c, ok := q.Criteria.(NoulCriteria)
	if !ok {
		t.Fatalf("criteria type: want NoulCriteria, got %T", q.Criteria)
	}
	if c.True != "yes it's urgent" || c.False != "not urgent" {
		t.Errorf("criteria: got %+v", c)
	}
}

func TestNewDecisionsProviderFactory(t *testing.T) {
	p := NewDecisionsProvider("typesafe", "key", "")
	if p == nil {
		t.Fatal("want non-nil provider")
	}
	if NewDecisionsProvider("unknown", "key", "") != nil {
		t.Fatal("want nil for unknown provider")
	}
}
