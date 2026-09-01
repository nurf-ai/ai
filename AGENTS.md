# AGENTS.md — ai

Multi-provider LLM library. Single `package ai`, module `github.com/nurf-ai/ai`.

## Files

| File | What |
|------|------|
| `ai_contract.go` | `LLMProvider` interface, `ModelFunc`, `Middleware`, `Chain` |
| `anthropic.go` | Anthropic provider |
| `openai.go` | OpenAI provider |
| `gemini.go` | Google Gemini provider |
| `ollama.go` | Ollama provider (local models) |
| `huggingface.go` | Hugging Face Inference API |
| `gemini_image_provider.go` | Gemini image generation |
| `openai_image_provider.go` | OpenAI image generation/editing |
| `openai_stt_provider.go` | OpenAI speech-to-text |
| `fal.go` | fal.ai queue client (`FalClient`: submit / status / result / cancel / run) |
| `fal_video_provider.go` | fal video generation (LTX-2.3 image/text-to-video) |
| `video_provider.go` | `VideoProvider` interface, `VideoRequest` / `VideoResult`, `DataURI` |
| `openai_moderation_provider.go` | OpenAI content moderation |
| `image_provider.go` | `ImageProvider` interface |
| `stt_provider.go` | `STTProvider` interface |
| `moderation_provider.go` | `ModerationProvider` interface |
| `meter.go` | Usage metering hooks, prompt block attribution |
| `models.go` | Cost estimation (`EstimateCostFull`, `EstimateVideoCost`) + context window lookup, loads `models.json` via `go:embed` |
| `models.json` | Per-model pricing + context windows (single source of truth) |
| `model_limits.go` | `MaxInputTokensLLM` helper, token counting |
| `provider.go` | Provider registry, `NewLLMProvider` factory |
| `types.go` | `Message`, `Tool`, `Response`, shared types |
| `parts.go` | `StructuredOutputFromParts`, `MultimodalStructuredProvider`, multimodal helpers |
| `stream.go` | `Stream`, `StreamWithChan`, `StreamChunk`, `StreamResult`, `StreamingProvider` |
| `stream_openai.go` | OpenAI SSE streaming loop (internal) |
| `errors.go` | Sentinel errors |
| `logger.go` | Package-level zap logger |

## Contributing

### Adding a provider

1. Create `<provider>.go` implementing `LLMProvider`
2. Add model entries to `models.json` (pricing + `max_input_tokens`)
3. Register in `provider.go` (`NewLLMProvider` switch)
4. Add a smoke test to `integration_test.go`
5. Run both test suites before submitting

### Adding a model

1. Add entry to `models.json` under the provider key — include `max_input_tokens`
2. `go test ./...` — `TestPricingCoverage` will fail if `max_input_tokens` is missing

Provider model catalogs:
- Anthropic: https://docs.anthropic.com/en/docs/about-claude/models
- OpenAI: https://platform.openai.com/docs/models
- Gemini: https://ai.google.dev/gemini-api/docs/models
- Hugging Face: https://huggingface.co/models
- Ollama: https://ollama.com/library

### Testing

Unit tests (no API keys needed, runs in CI):

```bash
go test ./...
```

Integration tests (real API calls, run locally):

```bash
cp .env.test.tpl .env.test   # fill in your keys
go test -tags=integration -count=1 -json ./... | go run ./cmd/testmatrix
```

`.env.test` is loaded automatically via `godotenv` in `TestMain`. Only set the keys you have — providers with missing keys are skipped (`∅`).

`testmatrix` prints live progress with per-test token counts + cost, failure details, and per-provider totals to stderr; coverage matrix to stdout. Paste the matrix into README between `<!-- testmatrix:start/end -->` markers. PRs that change provider capabilities must include an updated matrix.

Split by provider with `-run`:

```bash
go test -tags=integration -json -run TestAnthropic ./... | go run ./cmd/testmatrix
```

See `.env.test.tpl` for the full list of env vars.

### Commits & releases

Use conventional commits (`feat:`, `fix:`, etc.). Release Please creates a release PR on push to main — merging it tags and publishes.
