package ai

import (
	"context"
	"fmt"
	"strings"

	"go.uber.org/zap"
)

// ModerationResult holds the outcome of a moderation check.
type ModerationResult struct {
	Flagged    bool            `json:"flagged"`
	Categories map[string]bool `json:"categories,omitempty"`
}

// ModerationError is returned when a prompt is blocked by moderation.
type ModerationError struct {
	Categories map[string]bool
}

func (e *ModerationError) Error() string {
	cats := make([]string, 0, len(e.Categories))
	for k := range e.Categories {
		cats = append(cats, k)
	}
	return fmt.Sprintf("blocked by moderation: %s", strings.Join(cats, ", "))
}

// ModerationProvider checks user-supplied text for policy violations.
type ModerationProvider interface {
	Check(ctx context.Context, input string) (*ModerationResult, error)
}

// NewModerationProvider creates a ModerationProvider from a provider name.
func NewModerationProvider(providerName, apiKey string) ModerationProvider {
	switch providerName {
	case "openai":
		return NewOpenAIModerationProvider(apiKey)
	default:
		return nil
	}
}

var onModerationError func(ctx context.Context, err error) error

// SetOnModerationError overrides the default moderation-error behavior.
// Default (nil): warn and allow. When set, the function's return value
// determines whether the call proceeds (nil) or is rejected (non-nil error).
func SetOnModerationError(fn func(ctx context.Context, err error) error) {
	onModerationError = fn
}

// checkModeration runs a moderation check if a provider is set.
// Returns a ModerationError if flagged, nil otherwise.
func checkModeration(ctx context.Context, m ModerationProvider, input string) error {
	if m == nil {
		return nil
	}
	res, err := m.Check(ctx, input)
	if err != nil {
		if onModerationError != nil {
			return onModerationError(ctx, err)
		}
		logger.Warn("check failed, allowing", zap.Error(err))
		return nil
	}
	if res.Flagged {
		return &ModerationError{Categories: res.Categories}
	}
	return nil
}
