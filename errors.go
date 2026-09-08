package ai

import (
	"errors"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	openai "github.com/sashabaranov/go-openai"
)

type ErrorKind int

const (
	ErrUnknown ErrorKind = iota
	ErrBilling
	ErrAuth
	ErrRateLimit
	ErrModeration
	ErrProviderDown
)

func ClassifyError(err error) (ErrorKind, string) {
	if err == nil {
		return ErrUnknown, ""
	}

	var modErr *ModerationError
	if errors.As(err, &modErr) {
		return ErrModeration, "message flagged by content filter"
	}

	var aErr *anthropic.Error
	if errors.As(err, &aErr) {
		return classifyHTTP(aErr.StatusCode, aErr.Error())
	}

	var oErr *openai.APIError
	if errors.As(err, &oErr) {
		return classifyHTTP(oErr.HTTPStatusCode, oErr.Error())
	}

	var fErr *FalError
	if errors.As(err, &fErr) {
		return classifyHTTP(fErr.Status, fErr.Message)
	}

	msg := err.Error()
	if strings.Contains(msg, "context canceled") || strings.Contains(msg, "context deadline exceeded") {
		return ErrUnknown, "request timed out"
	}

	return ErrUnknown, ""
}

// outOfCredits spots a spent account in a provider's message. Providers say it
// under whatever status suits them — OpenAI as a 429 ("You have no credits
// remaining", "insufficient_quota"), fal as a 403 ("User is locked. Reason:
// TOP_UP") — so the body decides, not the code. Getting this wrong is not
// cosmetic: a rate limit clears by waiting and a spent account never does, and
// everything downstream (retry, hold, what the viewer is told) turns on it.
func outOfCredits(lower string) bool {
	for _, sign := range []string{
		"no credits remaining", "insufficient_quota", "exceeded your current quota",
		"credit", "billing", "balance", "top_up", "user is locked",
	} {
		if strings.Contains(lower, sign) {
			return true
		}
	}
	return false
}

func classifyHTTP(status int, msg string) (ErrorKind, string) {
	lower := strings.ToLower(msg)

	switch {
	case status == 401:
		return ErrAuth, "AI provider API key is invalid or expired"
	case (status == 403 || status == 400 || status == 429) && outOfCredits(lower):
		return ErrBilling, "AI provider account needs credits — check billing"
	case status == 403:
		return ErrAuth, "AI provider denied access"
	case status == 429:
		return ErrRateLimit, "AI provider rate limit reached, try again shortly"
	case status >= 500:
		return ErrProviderDown, "AI provider is temporarily unavailable"
	default:
		return ErrUnknown, ""
	}
}

func (k ErrorKind) UserMessage() string {
	switch k {
	case ErrBilling:
		return "AI provider account needs credits — check billing"
	case ErrAuth:
		return "AI provider API key is invalid"
	case ErrRateLimit:
		return "AI provider rate limit reached, try again shortly"
	case ErrModeration:
		return "message flagged by content filter"
	case ErrProviderDown:
		return "AI provider is temporarily unavailable"
	default:
		return "something went wrong, please try again"
	}
}

func (k ErrorKind) Retryable() bool {
	return k == ErrRateLimit || k == ErrProviderDown
}

// ViewerMessage is what a person who cannot fix it should read. The provider
// account and its key are the platform's, not theirs and not the surf owner's,
// so the only thing worth telling them is whether waiting helps. Billing and
// auth deliberately collapse into one line: "spent" and "bad key" both mean
// the platform's setup is broken, and telling them apart is a free hint about
// somebody's secret. The true cause travels in the logs and in Code().
func (k ErrorKind) ViewerMessage() string {
	switch k {
	case ErrRateLimit, ErrProviderDown:
		return "the model is busy right now — try again in a moment"
	case ErrBilling, ErrAuth:
		return "the model is unavailable right now — this is not your connection, and waiting will not clear it"
	case ErrModeration:
		return "message flagged by content filter"
	default:
		return "something went wrong, please try again"
	}
}

// Code is the stable, greppable name of what happened: the field to switch on
// in a client, a log query or a dashboard, since the prose deliberately does
// not distinguish. `ai_unavailable` means stop retrying — a person has to fix
// an account somewhere; `ai_busy` clears on its own.
func (k ErrorKind) Code() string {
	switch k {
	case ErrRateLimit, ErrProviderDown:
		return "ai_busy"
	case ErrBilling, ErrAuth:
		return "ai_unavailable"
	case ErrModeration:
		return "ai_flagged"
	default:
		return "ai_error"
	}
}
