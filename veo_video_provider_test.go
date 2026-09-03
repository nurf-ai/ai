package ai

import "testing"

func TestVeoFit(t *testing.T) {
	cases := []struct {
		res     string
		dur     float64
		wantRes string
		wantDur int
	}{
		{"1080p", 6, "720p", 6},  // the tv station's request: keep 6 s, drop to 720p
		{"1080p", 8, "1080p", 8}, // 1080p is fine at 8 s
		{"1080P", 6, "720p", 6},
		{"720p", 6, "720p", 6},
		{"1080p", 0, "1080p", 0}, // duration unset → provider default (8 s) keeps 1080p
		{"720p", 10, "720p", 8},  // clamp to Veo's 4–8 s
		{"", 2, "", 4},
	}
	for _, c := range cases {
		gotRes, gotDur := veoFit(c.res, c.dur)
		if gotRes != c.wantRes || gotDur != c.wantDur {
			t.Errorf("veoFit(%q, %v) = (%q, %d), want (%q, %d)", c.res, c.dur, gotRes, gotDur, c.wantRes, c.wantDur)
		}
	}
}
