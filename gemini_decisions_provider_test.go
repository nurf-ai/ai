package ai

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestGeminiDecisionsJudge(t *testing.T) {
	description := "Billing issues"
	questions := map[string]DecisionsQuestion{
		"binary":   NoulWith("Is this urgent?", "Urgent", "Routine"),
		"category": Choice("Category?", map[string]*string{"billing": &description, "other": nil}),
		"rating":   Score("Severity?", []string{"Low", "High"}),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/decisions" || r.Header.Get("x-goog-api-key") != "key" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var body struct {
			Model     string
			Input     string
			Questions map[string]struct{ Criteria any }
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body.Model != "gemini-decision-1" || body.Input != `{"ticket":"urgent"}` {
			t.Errorf("body: %+v", body)
		}
		if !reflect.DeepEqual(body.Questions["binary"].Criteria, map[string]any{"true_description": "Urgent", "false_description": "Routine"}) {
			t.Errorf("binary: %+v", body.Questions["binary"])
		}
		if !reflect.DeepEqual(body.Questions["category"].Criteria, map[string]any{"billing": "Billing issues", "other": ""}) {
			t.Errorf("category: %+v", body.Questions["category"])
		}
		_, _ = w.Write([]byte(`{"answers":{"binary":{"type":"noul","noul":0.94,"confidence":0.94},"category":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"other":0.1},"confidence":0.9},"rating":{"type":"score","score":0.8,"legend":{"0":"Low","1":"High"},"probabilities":{"0":0.2,"1":0.8},"confidence":0.8}},"usage":{"total_input_tokens":324,"total_output_tokens":0,"total_tokens":324}}`))
	}))
	defer srv.Close()
	p := NewGeminiDecisionsProvider("key", WithGeminiDecisionsBase(srv.URL+"/"), WithGeminiDecisionsHTTPClient(srv.Client()))
	var events []UsageEvent
	SetDecisionsMeter(p, func(ev UsageEvent) { events = append(events, ev) })
	result, err := p.Judge(WithDebugSpanID(context.Background(), "span"), &DecisionsRequest{State: map[string]string{"ticket": "urgent"}, Questions: questions})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != "gemini-decision-1" || result.Usage.InputTokens != 324 || result.Usage.OutputTokens != 0 || len(result.Answers) != 3 {
		t.Fatalf("result: %+v", result)
	}
	if result.Answers["binary"].Confidence != 0.94 || result.Answers["rating"].Score != 0.8 {
		t.Fatalf("answers: %+v", result.Answers)
	}
	if len(events) != 1 {
		t.Fatalf("events: %v", events)
	}
	ev := events[0]
	if ev.Provider != "gemini" || ev.DebugSpanID != "span" || ev.InputTokens != 324 || ev.TotalTokens != 324 || math.Abs(ev.EstimatedCostUSD-0.0000324) > 1e-12 {
		t.Errorf("event: %+v", ev)
	}
	if questions["category"].Criteria.(map[string]*string)["other"] != nil {
		t.Error("mutated question")
	}
}

func TestGeminiDecisionsInput(t *testing.T) {
	for _, state := range []any{"hello", DecisionsContent{Type: "image", Data: "aGVsbG8=", MIMEType: "image/png"}, []DecisionsContent{{Type: "text", Text: "Look"}, {Type: "video", URI: "https://example.com/video.mp4", MIMEType: "video/mp4", Resolution: "low"}}} {
		t.Run(reflect.TypeOf(state).String(), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				want, _ := json.Marshal(state)
				if string(body["input"]) != string(want) {
					t.Errorf("input: %s, want %s", body["input"], want)
				}
				if string(body["model"]) != `"custom"` {
					t.Errorf("model: %s", body["model"])
				}
				_, _ = w.Write([]byte(`{"answers":{},"usage":{}}`))
			}))
			defer srv.Close()
			p := NewGeminiDecisionsProvider("key", WithGeminiDecisionsBase(srv.URL), WithGeminiDecisionsModel("custom"))
			if _, err := p.Judge(context.Background(), &DecisionsRequest{State: state, Questions: map[string]DecisionsQuestion{"q": Noul("yes?")}}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGeminiDecisionsErrors(t *testing.T) {
	for _, tt := range []struct {
		name, body string
		status     int
	}{
		{"safety", `{"error":{"code":"safety","message":"Blocked"}}`, 400},
		{"rate limit", `{"error":{"code":429,"message":"Quota exceeded"}}`, 429},
		{"plain", "unavailable", 503},
		{"malformed", "{", 200},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()
			p := NewGeminiDecisionsProvider("key", WithGeminiDecisionsBase(srv.URL))
			p.SetMeter(func(UsageEvent) { t.Error("metered failed request") })
			_, err := p.Judge(context.Background(), &DecisionsRequest{State: "hello"})
			if err == nil {
				t.Fatal("expected error")
			}
			if tt.status != 200 {
				var apiErr *GeminiDecisionsError
				if !errors.As(err, &apiErr) || apiErr.Status != tt.status {
					t.Fatalf("error: %v", err)
				}
			}
		})
	}
	p := NewGeminiDecisionsProvider("key")
	if _, err := p.Judge(context.Background(), nil); err == nil {
		t.Error("nil request accepted")
	}
	if _, err := p.Judge(context.Background(), &DecisionsRequest{State: make(chan int)}); err == nil {
		t.Error("unencodable state accepted")
	}
	if _, err := p.Judge(context.Background(), &DecisionsRequest{State: "hello", Questions: map[string]DecisionsQuestion{"q": Noul(make(chan int))}}); err == nil {
		t.Error("unencodable question accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Judge(ctx, &DecisionsRequest{State: "hello"}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancellation: %v", err)
	}
}

func TestGeminiDecisionsFactory(t *testing.T) {
	for _, model := range []string{"", "custom"} {
		p, ok := NewDecisionsProvider("gemini", "key", model).(*GeminiDecisionsProvider)
		if !ok {
			t.Fatal("wrong provider")
		}
		want := model
		if want == "" {
			want = "gemini-decision-1"
		}
		if p.model != want {
			t.Errorf("model: %s, want %s", p.model, want)
		}
	}
	if got, err := MaxInputTokensLLM("gemini", "gemini-decision-1"); err != nil || got != 262144 {
		t.Errorf("context: %d", got)
	}
}

func TestGeminiDecisionsJSONCriteria(t *testing.T) {
	var req DecisionsRequest
	if err := json.Unmarshal([]byte(`{"state":"hello","questions":{"q":{"type":"noul","instructions":"yes?","criteria":{"true":"Yes","false":"No"}},"c":{"type":"choice","instructions":"which?","criteria":{"a":null,"b":"Other"}}}}`), &req); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Questions map[string]struct{ Criteria map[string]any }
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if !reflect.DeepEqual(body.Questions["q"].Criteria, map[string]any{"true_description": "Yes", "false_description": "No"}) {
			t.Errorf("noul: %+v", body.Questions["q"])
		}
		if body.Questions["c"].Criteria["a"] != "" {
			t.Errorf("choice: %+v", body.Questions["c"])
		}
		_, _ = w.Write([]byte(`{"answers":{},"usage":{}}`))
	}))
	defer srv.Close()
	if _, err := NewGeminiDecisionsProvider("key", WithGeminiDecisionsBase(srv.URL)).Judge(context.Background(), &req); err != nil {
		t.Fatal(err)
	}
	if req.Questions["c"].Criteria.(map[string]any)["a"] != nil {
		t.Error("mutated criteria")
	}
}
