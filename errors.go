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

func classifyHTTP(status int, msg string) (ErrorKind, string) {
	lower := strings.ToLower(msg)

	switch {
	case status == 401:
		return ErrAuth, "AI provider API key is invalid or expired"
	case status == 403:
		return ErrAuth, "AI provider denied access"
	case status == 429:
		return ErrRateLimit, "AI provider rate limit reached, try again shortly"
	case status == 400 && (strings.Contains(lower, "credit") || strings.Contains(lower, "billing") || strings.Contains(lower, "balance")):
		return ErrBilling, "AI provider account needs credits — check billing"
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
