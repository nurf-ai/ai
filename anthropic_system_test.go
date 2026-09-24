package ai

import (
	"context"
	"testing"
)

// A caller that names no system prompt must not produce a system block:
// the Messages API rejects an empty text block ("system: text content blocks
// must be non-empty"), and omitting the field is the documented way to send
// no system prompt.
func TestAnthropicBuildSysBlocksOmitsEmpty(t *testing.T) {
	p := &AnthropicProvider{}
	if got := p.buildSysBlocks(context.Background(), ""); got != nil {
		t.Fatalf("empty system prompt must yield no blocks, got %v", got)
	}
	blocks := p.buildSysBlocks(context.Background(), "be terse")
	if len(blocks) != 1 || blocks[0].Text != "be terse" {
		t.Fatalf("a system prompt yields one text block, got %v", blocks)
	}
	if blocks[0].CacheControl.Type != "" {
		t.Fatalf("no cache_control unless WithCacheSysPrompt asks for it: %v", blocks[0].CacheControl)
	}
	cached := p.buildSysBlocks(WithCacheSysPrompt(context.Background()), "be terse")
	if cached[0].CacheControl.Type == "" {
		t.Fatal("WithCacheSysPrompt marks the block ephemeral")
	}
}
