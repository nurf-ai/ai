package ai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// Newer Claude models reject a thinking budget with a 400 ("thinking.type.enabled
// is not supported for this model"): they get adaptive thinking and an effort.
func TestSetAnthropicReasoning(t *testing.T) {
	for _, c := range []struct {
		model, effort string
		want, notWant []string
	}{
		{"claude-opus-5-5", "medium", []string{`"thinking":{"type":"adaptive"}`, `"effort":"medium"`}, []string{"budget_tokens"}},
		{"claude-sonnet-4-6", "minimal", []string{`"type":"adaptive"`, `"effort":"low"`}, []string{"budget_tokens"}},
		{"us.anthropic.claude-haiku-5-5", "xhigh", []string{`"type":"adaptive"`, `"effort":"xhigh"`}, nil},
		{"claude-sonnet-4-5-20250929", "high", []string{`"budget_tokens":16384`, `"type":"enabled"`}, []string{"effort"}},
		{"claude-3-7-sonnet-latest", "low", []string{`"budget_tokens":1024`}, []string{"adaptive"}},
		{"claude-opus-5-5", "", nil, []string{"thinking", "effort"}},
	} {
		p := anthropic.MessageNewParams{MaxTokens: 4096, Model: anthropic.Model(c.model)}
		setAnthropicReasoning(&p, c.model, c.effort)
		b, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		s := string(b)
		for _, w := range c.want {
			if !strings.Contains(s, w) {
				t.Errorf("%s/%s: want %s in %s", c.model, c.effort, w, s)
			}
		}
		for _, w := range c.notWant {
			if strings.Contains(s, w) {
				t.Errorf("%s/%s: did not want %s in %s", c.model, c.effort, w, s)
			}
		}
	}
}
