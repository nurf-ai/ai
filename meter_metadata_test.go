package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestWithMeterMetadata(t *testing.T) {
	tests := []struct {
		name   string
		stamps []map[string]any // applied in order
		want   map[string]any
	}{
		{name: "nil kv stamps nothing", stamps: []map[string]any{nil}, want: nil},
		{name: "empty kv stamps nothing", stamps: []map[string]any{{}}, want: nil},
		{name: "single stamp", stamps: []map[string]any{{"surf": "tv"}}, want: map[string]any{"surf": "tv"}},
		{
			name:   "later keys win, untouched keys kept",
			stamps: []map[string]any{{"surf": "tv", "session": 1}, {"surf": "home"}},
			want:   map[string]any{"surf": "home", "session": 1},
		},
		{
			name:   "empty stamp after a real one keeps it",
			stamps: []map[string]any{{"surf": "tv"}, nil, {}},
			want:   map[string]any{"surf": "tv"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			for _, kv := range tt.stamps {
				ctx = WithMeterMetadata(ctx, kv)
			}
			if got := MeterMetadataFromCtx(ctx); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MeterMetadataFromCtx = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestWithMeterMetadata_EmptyReturnsSameCtx(t *testing.T) {
	base := WithMeterOperation(context.Background(), "chat")
	for name, kv := range map[string]map[string]any{"nil": nil, "empty": {}} {
		if got := WithMeterMetadata(base, kv); got != base {
			t.Errorf("%s kv: got a new ctx, want the input ctx unchanged", name)
		}
	}
}

func TestWithMeterMetadata_CopySemantics(t *testing.T) {
	kv := map[string]any{"surf": "tv"}
	parent := WithMeterMetadata(context.Background(), kv)
	child := WithMeterMetadata(parent, map[string]any{"session": 1})

	// Caller mutates its map after stamping: ctx unaffected.
	kv["surf"] = "mutated"
	// Caller mutates the copy handed back: ctx unaffected.
	MeterMetadataFromCtx(parent)["injected"] = true

	want := map[string]any{"surf": "tv"}
	if got := MeterMetadataFromCtx(parent); !reflect.DeepEqual(got, want) {
		t.Errorf("parent metadata = %v, want %v", got, want)
	}
	// Stamping a child never leaks back into the parent.
	wantChild := map[string]any{"surf": "tv", "session": 1}
	if got := MeterMetadataFromCtx(child); !reflect.DeepEqual(got, wantChild) {
		t.Errorf("child metadata = %v, want %v", got, wantChild)
	}
}

func TestMeterMetadataFromCtx_Missing(t *testing.T) {
	if got := MeterMetadataFromCtx(context.Background()); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

func TestMergeMeterMetadata(t *testing.T) {
	tests := []struct {
		name string
		ctx  map[string]any // stamped on ctx; nil = nothing stamped
		md   map[string]any // provider-set
		want map[string]any
	}{
		{name: "both nil", want: nil},
		{name: "both empty", ctx: map[string]any{}, md: map[string]any{}, want: nil},
		{name: "ctx only, nil md", ctx: map[string]any{"surf": "tv"}, want: map[string]any{"surf": "tv"}},
		{name: "ctx only, empty md", ctx: map[string]any{"surf": "tv"}, md: map[string]any{}, want: map[string]any{"surf": "tv"}},
		{name: "md only", md: map[string]any{"type": "image_gen"}, want: map[string]any{"type": "image_gen"}},
		{
			name: "disjoint keys union",
			ctx:  map[string]any{"surf": "tv"},
			md:   map[string]any{"type": "image_gen"},
			want: map[string]any{"surf": "tv", "type": "image_gen"},
		},
		{
			name: "provider keys win on overlap",
			ctx:  map[string]any{"type": "caller", "surf": "tv"},
			md:   map[string]any{"type": "image_gen"},
			want: map[string]any{"type": "image_gen", "surf": "tv"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := WithMeterMetadata(context.Background(), tt.ctx)
			got := mergeMeterMetadata(ctx, tt.md)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mergeMeterMetadata = %v, want %v", got, tt.want)
			}
			if len(tt.ctx) == 0 {
				return
			}
			// The merge must neither mutate the ctx map nor alias it.
			if got != nil {
				got["mutated"] = true
			}
			if stamped := MeterMetadataFromCtx(ctx); !reflect.DeepEqual(stamped, tt.ctx) {
				t.Errorf("ctx metadata changed by merge: %v, want %v", stamped, tt.ctx)
			}
		})
	}
}

// TestOllamaProvider_MeterMetadataEndToEnd drives a real Chat call through an
// httptest server and asserts ctx-stamped metadata reaches the emitted
// UsageEvent, with the provider-set "blocks" key winning over a stamped one.
func TestOllamaProvider_MeterMetadataEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "x", "object": "chat.completion", "model": "m",
			"choices": []map[string]any{{
				"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": "hi"},
			}},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	}))
	defer srv.Close()

	p := NewOllamaProvider(srv.URL+"/v1", "m")
	var events []UsageEvent
	p.SetMeter(func(ev UsageEvent) { events = append(events, ev) })

	ctx := WithMeterMetadata(context.Background(), map[string]any{"surf": "tv", "blocks": "caller"})
	ctx = WithPromptBlocks(ctx, map[string]string{"system": "be brief"})
	if _, err := p.Chat(ctx, []Message{{Role: RoleUser, Content: "hello"}}, nil); err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("got %d usage events, want 1", len(events))
	}
	md := events[0].Metadata
	if md["surf"] != "tv" {
		t.Errorf(`Metadata["surf"] = %v, want "tv"`, md["surf"])
	}
	if _, ok := md["blocks"].(map[string]BlockSize); !ok {
		t.Errorf(`Metadata["blocks"] = %T, want provider blocks to win over the stamped value`, md["blocks"])
	}
}
