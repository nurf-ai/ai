package ai

import (
	"errors"
	"strings"
	"time"

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
	// ErrInvalidInput: the model can't take the request as given (too long,
	// too short, a field it doesn't know). Unlike the others it is the
	// caller's to fix, and the detail says how.
	ErrInvalidInput
	// ErrQuota: the application's own spending limit for this caller is
	// reached. Nothing went to a provider and no provider account is
	// involved; the limit lifts when it resets, and the detail says when.
	ErrQuota
)

// QuotaError is what an application returns, before a call is sent, when
// the caller's own spending limit is reached. Msg is written for the caller
// and safe to show them; RetryAfter is how long until the limit resets (0
// when unknown).
type QuotaError struct {
	Msg        string
	RetryAfter time.Duration
}

func (e *QuotaError) Error() string { return e.Msg }

// InputError is a request a model refuses as given. Providers return it
// before anything is sent (or billed); Field names the input, Msg says what
// to change.
type InputError struct {
	Model string
	Field string
	Msg   string
}

func (e *InputError) Error() string {
	if e.Field == "" {
		return e.Model + ": " + e.Msg
	}
	return e.Model + ": " + e.Field + " " + e.Msg
}

func ClassifyError(err error) (ErrorKind, string) {
	if err == nil {
		return ErrUnknown, ""
	}

	var modErr *ModerationError
	if errors.As(err, &modErr) {
		return ErrModeration, "message flagged by content filter"
	}

	var inErr *InputError
	if errors.As(err, &inErr) {
		return ErrInvalidInput, inErr.Error()
	}

	var qErr *QuotaError
	if errors.As(err, &qErr) {
		return ErrQuota, qErr.Msg
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
		// fal answers a request its input schema rejects with 400/422 and the
		// field-level reasons; that is the caller's to fix, not an outage.
		if (fErr.Status == 400 || fErr.Status == 422) && !outOfCredits(strings.ToLower(fErr.Message)) {
			return ErrInvalidInput, fErr.Endpoint + ": " + fErr.Message
		}
		return classifyHTTP(fErr.Status, fErr.Message)
	}

	var tErr *TypesafeError
	if errors.As(err, &tErr) {
		return classifyHTTP(tErr.Status, tErr.Message)
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
	case ErrInvalidInput:
		return "the model can't take this input as given; change it and try again"
	case ErrQuota:
		return "usage limit reached; it lifts when the limit resets"
	default:
		return "something went wrong, please try again"
	}
}

func (k ErrorKind) Retryable() bool {
	return k == ErrRateLimit || k == ErrProviderDown
}

// ViewerMessage is what a person who cannot fix it should read. The provider
// account and its key belong to whoever runs the service, not to the person
// reading, so the only thing worth telling them is whether waiting helps. Billing and
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
	case ErrInvalidInput:
		return "the model can't take this input as given; change it and try again"
	case ErrQuota:
		return "you've reached your usage limit for now; it lifts when the limit resets"
	default:
		return "something went wrong, please try again"
	}
}

// ViewerMessageFor is what a person who cannot fix err should read: the
// application's own words for a QuotaError (it wrote them for that person,
// with when the limit resets), the kind's ViewerMessage otherwise.
func ViewerMessageFor(err error) string {
	var qErr *QuotaError
	if errors.As(err, &qErr) && qErr.Msg != "" {
		return qErr.Msg
	}
	kind, _ := ClassifyError(err)
	return kind.ViewerMessage()
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
	case ErrInvalidInput:
		return "ai_invalid_input"
	case ErrQuota:
		return "ai_quota"
	default:
		return "ai_error"
	}
}
