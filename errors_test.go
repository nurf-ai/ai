package ai

import "testing"

// A spent account and a rate limit arrive under the same status from some
// providers, and mean opposite things: one clears by waiting, the other never
// does. The body decides.
func TestClassify_SpentAccountIsNotARateLimit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		msg    string
		want   ErrorKind
	}{
		{"openai 429 out of credits", 429, "You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.", ErrBilling},
		{"openai 429 quota", 429, "You exceeded your current quota, please check your plan and billing details", ErrBilling},
		{"openai insufficient_quota code", 429, "error, status code: 429, type: insufficient_quota", ErrBilling},
		{"a real rate limit", 429, "Rate limit reached for gpt-5 in organization org-x on tokens per min", ErrRateLimit},
		{"fal balance lock", 403, "User is locked. Reason: TOP_UP", ErrBilling},
		{"a plain 403", 403, "forbidden", ErrAuth},
		{"bad key", 401, "Incorrect API key provided", ErrAuth},
		{"provider down", 503, "upstream unavailable", ErrProviderDown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, msg := classifyHTTP(tc.status, tc.msg)
			if got != tc.want {
				t.Fatalf("kind = %v, want %v (%q)", got, tc.want, msg)
			}
			// Retryable must follow the kind: waiting only ever helps a
			// transient one, and a worker that retries a spent account spins
			// forever.
			if want := tc.want == ErrRateLimit || tc.want == ErrProviderDown; got.Retryable() != want {
				t.Fatalf("Retryable() = %v, want %v", got.Retryable(), want)
			}
		})
	}
}

// A viewer is told whether waiting helps and nothing else; billing and auth
// must be indistinguishable to them, and neither may name money or a key.
func TestViewerMessage_TellsRetryAndNothingElse(t *testing.T) {
	if ErrBilling.ViewerMessage() != ErrAuth.ViewerMessage() {
		t.Fatal("a viewer must not be able to tell a spent account from a bad key")
	}
	for _, k := range []ErrorKind{ErrBilling, ErrAuth, ErrRateLimit, ErrProviderDown} {
		msg := k.ViewerMessage()
		for _, leak := range []string{"credit", "billing", "key", "quota", "openai", "anthropic"} {
			if containsFold(msg, leak) {
				t.Fatalf("%v leaks %q to a viewer: %q", k, leak, msg)
			}
		}
	}
	if ErrBilling.Code() != ErrAuth.Code() || ErrBilling.Code() != "ai_unavailable" {
		t.Fatalf("codes: billing=%q auth=%q", ErrBilling.Code(), ErrAuth.Code())
	}
	if ErrRateLimit.Code() != "ai_busy" {
		t.Fatalf("rate limit code = %q", ErrRateLimit.Code())
	}
	// The operator's line still says the true thing — that is the whole point
	// of keeping two.
	if !containsFold(ErrBilling.UserMessage(), "credits") {
		t.Fatalf("operator message = %q", ErrBilling.UserMessage())
	}
}

func containsFold(s, sub string) bool {
	return len(sub) > 0 && len(s) >= len(sub) && indexFold(s, sub) >= 0
}

func indexFold(s, sub string) int {
	ls, lsub := lower(s), lower(sub)
	for i := 0; i+len(lsub) <= len(ls); i++ {
		if ls[i:i+len(lsub)] == lsub {
			return i
		}
	}
	return -1
}

func lower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}
