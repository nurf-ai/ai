// testmatrix reads `go test -json` output from stdin, prints live progress
// to stderr as results arrive, then prints the final matrix to stdout.
// Failed test output is printed after the summary for debugging.
// Cost lines (##cost##) emitted by integration tests are aggregated per provider.
//
// Usage:
//
//	go test -tags=integration -json ./... | go run ./cmd/testmatrix
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"
)

const (
	cLabel  = "\033[38;5;60m"  // dark blue — tok_in, tok_out, tok_cached
	cCount  = "\033[38;5;75m"  // light blue — token counts
	cDollar = "\033[38;5;28m"  // dark green — $ sign
	cAmount = "\033[38;5;71m"  // light green — dollar amount
	cReset  = "\033[0m"
	costCol = 68
)

type testEvent struct {
	Action  string  `json:"Action"`
	Test    string  `json:"Test"`
	Output  string  `json:"Output"`
	Elapsed float64 `json:"Elapsed"`
}

type column struct {
	header  string
	subtest string
}

var columns = []column{
	{"Chat", "Chat"},
	{"Stream", "Stream"},
	{"Reasoning", "Reasoning"},
	{"Structured Output", "StructuredOutput"},
	{"From Schema", "StructuredOutputFromSchema"},
	{"Tools", "ChatWithTools"},
	{"Embeddings", "Embeddings"},
	{"STT", "STT"},
	{"Moderation", "Moderation"},
	{"Image Gen", "ImageGenerate"},
	{"Img Edit", "ImageEdit"},
	{"Img Edit Ref", "ImageEditWithReference"},
	{"Txt2Vid", "TextToVideo"},
	{"Img2Vid", "VideoGenerate"},
	{"Caching", "PromptCaching"},
}

var leafSubtests = func() map[string]bool {
	m := map[string]bool{}
	for _, c := range columns {
		m[c.subtest] = true
	}
	m["StreamWithChan"] = true
	m["VideoGenerate_MinimaxH3Max"] = true
	return m
}()

type row struct {
	provider   string
	model      string
	testPrefix string
	only       map[string]bool
	alias      map[string]string // column subtest → actual test subtest
}

var chatCaps = map[string]bool{
	"Chat": true, "Stream": true, "StreamWithChan": true,
	"StructuredOutput": true, "StructuredOutputFromSchema": true,
	"ChatWithTools": true,
}

var chatReasonCaps = map[string]bool{
	"Chat": true, "Stream": true, "StreamWithChan": true,
	"Reasoning": true, "StructuredOutput": true,
	"StructuredOutputFromSchema": true, "ChatWithTools": true,
	"PromptCaching": true,
}

var rows = []row{
	{provider: "Anthropic", model: "claude-haiku-4-5", testPrefix: "TestAnthropicChat_Integration", only: chatReasonCaps},
	{provider: "OpenAI", model: "gpt-4o-mini", testPrefix: "TestOpenAI_Integration", only: chatCaps},
	{provider: "OpenAI", model: "gpt-5-mini", testPrefix: "TestOpenAI_Integration", only: map[string]bool{"Reasoning": true}},
	{provider: "OpenAI", model: "text-embedding-3-small", testPrefix: "TestOpenAI_Integration", only: map[string]bool{"Embeddings": true}},
	{provider: "OpenAI", model: "whisper-1", testPrefix: "TestOpenAI_Integration", only: map[string]bool{"STT": true}},
	{provider: "OpenAI", model: "omni-moderation-latest", testPrefix: "TestOpenAI_Integration", only: map[string]bool{"Moderation": true}},
	{provider: "OpenAI", model: "gpt-image-1", testPrefix: "TestOpenAI_Integration", only: map[string]bool{"ImageGenerate": true, "ImageEdit": true}},
	{provider: "Gemini", model: "gemini-3.6-flash", testPrefix: "TestGemini_Integration", only: chatCaps},
	{provider: "Gemini", model: "gemini-2.5-flash-image", testPrefix: "TestGemini_Integration", only: map[string]bool{"ImageGenerate": true, "ImageEdit": true, "ImageEditWithReference": true}},
	{provider: "Hugging Face", model: "Kimi-K2-Instruct", testPrefix: "TestHuggingFace_Integration/moonshotai/Kimi-K2-Instruct-0905"},
	{provider: "Hugging Face", model: "Kimi-K3", testPrefix: "TestHuggingFace_Integration/moonshotai/Kimi-K3"},
	{provider: "Ollama", model: "qwen3.5:0.8b", testPrefix: "TestOllama_Integration/qwen3.5:0.8b"},
	{provider: "Ollama", model: "gpt-oss:20b", testPrefix: "TestOllama_Integration/gpt-oss:20b"},
	{provider: "Ollama", model: "gemma4:e4b", testPrefix: "TestOllama_Integration/gemma4:e4b"},
	{provider: "fal", model: "ltx-2.3/t2v/fast", testPrefix: "TestFal_Integration", only: map[string]bool{"TextToVideo": true}},
	{provider: "fal", model: "ltx-2.3/i2v/fast", testPrefix: "TestFal_Integration", only: map[string]bool{"VideoGenerate": true}},
	{provider: "fal", model: "minimax/h3-max/i2v", testPrefix: "TestFal_Integration", only: map[string]bool{"VideoGenerate": true}, alias: map[string]string{"VideoGenerate": "VideoGenerate_MinimaxH3Max"}},
}

var parentToProvider = map[string]string{
	"TestAnthropicChat_Integration": "Anthropic",
	"TestOpenAI_Integration":        "OpenAI",
	"TestGemini_Integration":        "Gemini",
	"TestHuggingFace_Integration":   "HuggingFace",
	"TestOllama_Integration":        "Ollama",
	"TestFal_Integration":           "fal",
}

func shortLabel(test string) string {
	if label, ok := parentToProvider[test]; ok {
		return label
	}
	for _, r := range rows {
		if rest, ok := strings.CutPrefix(test, r.testPrefix+"/"); ok {
			return r.provider + "/" + r.model + "/" + rest
		}
	}
	return test
}

func isLeaf(test string) bool {
	idx := strings.LastIndex(test, "/")
	if idx < 0 {
		return false
	}
	return leafSubtests[test[idx+1:]]
}

type providerCost struct {
	usd    float64
	in     int
	out    int
	cached int
}

func parseCostLine(line string) (usd float64, in, out, cached int, ok bool) {
	idx := strings.Index(line, "##cost##")
	if idx < 0 {
		return
	}
	ok = true
	_, _ = fmt.Sscanf(line[idx:], "##cost## usd=%f in=%d out=%d cached=%d", &usd, &in, &out, &cached)
	return
}

func main() {
	results := map[string]string{}
	skippedParents := map[string]bool{}
	output := map[string]*strings.Builder{}
	costs := map[string]*providerCost{}
	testCosts := map[string]*providerCost{}
	var failures []string
	var passed, failed, skipped int

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		var ev testEvent
		if json.Unmarshal(scanner.Bytes(), &ev) != nil {
			continue
		}
		if ev.Test == "" {
			continue
		}
		switch ev.Action {
		case "output":
			if output[ev.Test] == nil {
				output[ev.Test] = &strings.Builder{}
			}
			output[ev.Test].WriteString(ev.Output)

			if usd, in, out, cached, ok := parseCostLine(ev.Output); ok {
				provider := providerForTest(ev.Test)
				if provider != "" {
					if costs[provider] == nil {
						costs[provider] = &providerCost{}
					}
					costs[provider].usd += usd
					costs[provider].in += in
					costs[provider].out += out
					costs[provider].cached += cached
				}
				if testCosts[ev.Test] == nil {
					testCosts[ev.Test] = &providerCost{}
				}
				testCosts[ev.Test].usd += usd
				testCosts[ev.Test].in += in
				testCosts[ev.Test].out += out
				testCosts[ev.Test].cached += cached
			}
		case "pass":
			results[ev.Test] = ev.Action
			if isLeaf(ev.Test) {
				passed++
				printWithCost(fmt.Sprintf("  ✓ %s (%.1fs)", shortLabel(ev.Test), ev.Elapsed), testCosts[ev.Test])
			}
			delete(output, ev.Test)
		case "fail":
			results[ev.Test] = ev.Action
			if isLeaf(ev.Test) {
				failed++
				failures = append(failures, ev.Test)
				printWithCost(fmt.Sprintf("  ✗ %s (%.1fs)", shortLabel(ev.Test), ev.Elapsed), testCosts[ev.Test])
			}
		case "skip":
			results[ev.Test] = ev.Action
			skippedParents[ev.Test] = true
			if _, isParent := parentToProvider[ev.Test]; isParent {
				skipped++
				fmt.Fprintf(os.Stderr, "  ∅ %s\n", shortLabel(ev.Test))
			}
			delete(output, ev.Test)
		}
	}

	fmt.Fprintf(os.Stderr, "\n%d passed, %d failed, %d skipped\n", passed, failed, skipped)

	if len(failures) > 0 {
		fmt.Fprintf(os.Stderr, "\n── failures ──\n")
		for _, test := range failures {
			fmt.Fprintf(os.Stderr, "\n--- %s ---\n", shortLabel(test))
			if buf, ok := output[test]; ok {
				fmt.Fprint(os.Stderr, buf.String())
			}
		}
		fmt.Fprintln(os.Stderr)
	}

	if len(costs) > 0 {
		fmt.Fprintf(os.Stderr, "\n── cost ──\n")
		var totalUSD float64
		var totalIn, totalOut, totalCached int
		for _, provider := range []string{"Anthropic", "OpenAI", "Gemini", "HuggingFace", "Ollama"} {
			c, ok := costs[provider]
			if !ok {
				continue
			}
			fmt.Fprintf(os.Stderr, "  %-14s %s$%s%s%.4f%s  (%s%d%s %stok_in%s, %s%d%s %stok_out%s%s)\n",
				provider,
				cDollar, cReset, cAmount, c.usd, cReset,
				cCount, c.in, cReset, cLabel, cReset,
				cCount, c.out, cReset, cLabel, cReset,
				cachedSuffix(c.cached))
			totalUSD += c.usd
			totalIn += c.in
			totalOut += c.out
			totalCached += c.cached
		}
		fmt.Fprintf(os.Stderr, "  %-14s %s$%s%s%.4f%s  (%s%d%s %stok_in%s, %s%d%s %stok_out%s%s)\n",
			"TOTAL",
			cDollar, cReset, cAmount, totalUSD, cReset,
			cCount, totalIn, cReset, cLabel, cReset,
			cCount, totalOut, cReset, cLabel, cReset,
			cachedSuffix(totalCached))
	}
	fmt.Fprintln(os.Stderr)

	// matrix to stdout
	var sb strings.Builder

	fmt.Fprintf(&sb, "| Provider | Model |")
	for _, c := range columns {
		fmt.Fprintf(&sb, " %s |", c.header)
	}
	sb.WriteString("\n")

	sb.WriteString("|----------|-------|")
	for range columns {
		sb.WriteString(":---:|")
	}
	sb.WriteString("\n")

	for _, r := range rows {
		fmt.Fprintf(&sb, "| %s | `%s` |", r.provider, r.model)

		parentSkipped := skippedParents[r.testPrefix]
		if !parentSkipped {
			parts := strings.SplitN(r.testPrefix, "/", 2)
			parentSkipped = skippedParents[parts[0]]
		}

		for _, c := range columns {
			if r.only != nil && !r.only[c.subtest] {
				sb.WriteString(" |")
				continue
			}
			sub := c.subtest
			if a, ok := r.alias[sub]; ok {
				sub = a
			}
			key := r.testPrefix + "/" + sub
			switch results[key] {
			case "pass":
				sb.WriteString(" ✓ |")
			case "fail":
				sb.WriteString(" ✗ |")
			case "skip":
				sb.WriteString(" skip |")
			default:
				if parentSkipped {
					sb.WriteString(" skip |")
				} else {
					sb.WriteString(" |")
				}
			}
		}
		sb.WriteString("\n")
	}

	fmt.Print(sb.String())
}

func printWithCost(label string, c *providerCost) {
	cost := costSuffix(c)
	if cost == "" {
		fmt.Fprintln(os.Stderr, label)
		return
	}
	pad := costCol - utf8.RuneCountInString(label)
	if pad < 2 {
		pad = 2
	}
	fmt.Fprintf(os.Stderr, "%s%*s%s\n", label, pad, "", cost)
}

func costSuffix(c *providerCost) string {
	if c == nil || (c.in == 0 && c.out == 0) {
		return ""
	}
	return fmt.Sprintf("%s%d%s %stok_in%s, %s%d%s %stok_out%s, %s$%s%s%.4f%s",
		cCount, c.in, cReset, cLabel, cReset,
		cCount, c.out, cReset, cLabel, cReset,
		cDollar, cReset, cAmount, c.usd, cReset)
}

func cachedSuffix(cached int) string {
	if cached == 0 {
		return ""
	}
	return fmt.Sprintf(", %s%d%s %stok_cached%s", cCount, cached, cReset, cLabel, cReset)
}

func providerForTest(test string) string {
	for parent, provider := range parentToProvider {
		if test == parent || strings.HasPrefix(test, parent+"/") {
			return provider
		}
	}
	return ""
}
