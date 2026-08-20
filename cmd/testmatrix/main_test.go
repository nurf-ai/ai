package main

import (
	"strings"
	"testing"
)

func TestColumnSubtestsUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range columns {
		if seen[c.subtest] {
			t.Errorf("duplicate column subtest: %q", c.subtest)
		}
		seen[c.subtest] = true
	}
}

func TestRowCapsReferenceKnownSubtests(t *testing.T) {
	for _, r := range rows {
		if r.only == nil {
			continue
		}
		for sub := range r.only {
			if !leafSubtests[sub] {
				t.Errorf("row %s/%s: cap %q is not a known leaf subtest", r.provider, r.model, sub)
			}
		}
	}
}

func TestRowPrefixesMatchParentProviders(t *testing.T) {
	for _, r := range rows {
		found := false
		for parent := range parentToProvider {
			if r.testPrefix == parent || strings.HasPrefix(r.testPrefix, parent+"/") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("row %s/%s: testPrefix %q has no matching parentToProvider entry", r.provider, r.model, r.testPrefix)
		}
	}
}
