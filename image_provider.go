package ai

import "context"

// ImageProvider generates and edits images via an AI model.
type ImageProvider interface {
	Generate(ctx context.Context, prompt, model, size string) (string, error)                                 // base64
	Edit(ctx context.Context, image []byte, editPrompt string) (string, error)                                // base64
	EditWithReference(ctx context.Context, image []byte, reference []byte, editPrompt string) (string, error) // base64
}
