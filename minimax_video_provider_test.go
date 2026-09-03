package ai

import "testing"

func TestMinimaxResolution(t *testing.T) {
	cases := map[string]string{"": "", "768P": "768P", "768p": "768P", "1080p": "768P", "1080P": "768P", "720p": "480P", "1440p": "2K", "2K": "2K", "4k": "480P", "weird": "WEIRD"}
	for in, want := range cases {
		if got := minimaxResolution(in); got != want {
			t.Errorf("minimaxResolution(%q) = %q, want %q", in, got, want)
		}
	}
}
