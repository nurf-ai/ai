package ai

import (
	"context"
	"testing"

	"github.com/google/uuid"
)

func TestWithMeterCallerID(t *testing.T) {
	id := uuid.New()
	ctx := WithMeterCallerID(context.Background(), id)
	got := MeterCallerIDFromCtx(ctx)
	if got != id {
		t.Errorf("got %v, want %v", got, id)
	}
}

func TestMeterCallerIDFromCtx_Missing(t *testing.T) {
	got := MeterCallerIDFromCtx(context.Background())
	if got != uuid.Nil {
		t.Errorf("got %v, want uuid.Nil", got)
	}
}

func TestWithMeterOperation(t *testing.T) {
	ctx := WithMeterOperation(context.Background(), "chat")
	got := MeterOperationFromCtx(ctx)
	if got != "chat" {
		t.Errorf("got %q, want %q", got, "chat")
	}
}

func TestMeterOperationFromCtx_Missing(t *testing.T) {
	got := MeterOperationFromCtx(context.Background())
	if got != "unknown" {
		t.Errorf("got %q, want %q", got, "unknown")
	}
}

func TestAttachBlocks(t *testing.T) {
	ctx := WithPromptBlocks(context.Background(), map[string]string{
		"system": "you are helpful",
	})
	ev := &UsageEvent{}
	attachBlocks(ctx, ev)
	blocks, ok := ev.Metadata["blocks"]
	if !ok {
		t.Fatal("expected blocks in metadata")
	}
	bm := blocks.(map[string]BlockSize)
	if bm["system"].Chars != len("you are helpful") {
		t.Errorf("chars = %d, want %d", bm["system"].Chars, len("you are helpful"))
	}
}

func TestAttachBlocks_Empty(t *testing.T) {
	ev := &UsageEvent{}
	attachBlocks(context.Background(), ev)
	if ev.Metadata != nil {
		t.Error("expected nil metadata when no blocks")
	}
}
