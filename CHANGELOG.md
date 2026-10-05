# Changelog

## [0.17.0](https://github.com/nurf-ai/ai/compare/v0.16.0...v0.17.0) (2026-10-05)


### Features

* meter embeddings and speech-to-text; realtime usage carries its caller ([46e401a](https://github.com/nurf-ai/ai/commit/46e401aaf9ce7746badffca614d8109d79879494))
* QuotaError for an application's own spending limit ([3cff760](https://github.com/nurf-ai/ai/commit/3cff760d99865cf61228b0751c60f59b78bbfda9))
* ViewerMessageFor shows an application's quota message as written ([2e10114](https://github.com/nurf-ai/ai/commit/2e10114e4579762bbcede340766956d0fb0a5ddf))


### Bug Fixes

* a stream cut short still reports its usage ([edb6275](https://github.com/nurf-ai/ai/commit/edb6275c13c5ff59a1d3a35d37c79a69ca3067b8))
* price gemini-3.1-flash-lite-image and gpt-realtime-2.1 ([9bd8e19](https://github.com/nurf-ai/ai/commit/9bd8e197dc40513002feb16c50de252b720d85c9))

## [0.16.0](https://github.com/nurf-ai/ai/compare/v0.15.0...v0.16.0) (2026-10-04)


### Features

* say why a reply stopped; set aside a tool call the output cap cut off ([49df02c](https://github.com/nurf-ai/ai/commit/49df02c654b0c62dbc9d2be657098e92fd6a116a))

## [0.15.0](https://github.com/nurf-ai/ai/compare/v0.14.0...v0.15.0) (2026-10-01)


### Features

* music cover art, provider error wrapping, meter fixes ([3dfdfa2](https://github.com/nurf-ai/ai/commit/3dfdfa2a1849a86b7a653ecfc6aaf3c3f34c7958))
* MusicProvider interface + fal implementation ([4bc749b](https://github.com/nurf-ai/ai/commit/4bc749b79c6169b736f90fdef90fd545ba3413a2))

## [0.14.0](https://github.com/nurf-ai/ai/compare/v0.13.0...v0.14.0) (2026-09-29)


### Features

* **models:** add claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-1 ([9d4b8c9](https://github.com/nurf-ai/ai/commit/9d4b8c904b115a314a9621b2d84d5fe4f7500d52))


### Bug Fixes

* **anthropic:** send {} input for zero-arg tool calls ([fa4ea88](https://github.com/nurf-ai/ai/commit/fa4ea88a36a639443f726148b995158eaa2c638f))
* **anthropic:** structured output on models that reject forced tool_choice ([cac6133](https://github.com/nurf-ai/ai/commit/cac61334e65c87d69cd66ddec29d8ecfbf0dca1e))

## [0.13.0](https://github.com/nurf-ai/ai/compare/v0.12.0...v0.13.0) (2026-09-29)


### Features

* configurable system one provider, fal+openrouter image providers ([52e2e91](https://github.com/nurf-ai/ai/commit/52e2e914ef2149efb5f9ccf34c18007a50e3f8c6))

## [0.12.0](https://github.com/nurf-ai/ai/compare/v0.11.2...v0.12.0) (2026-09-29)


### Features

* **image:** add fal and OpenRouter image generation providers ([b14cc1b](https://github.com/nurf-ai/ai/commit/b14cc1b4d088063daf016e63e76d624eb5ae5077))

## [0.11.2](https://github.com/nurf-ai/ai/compare/v0.11.1...v0.11.2) (2026-09-29)


### Bug Fixes

* **meter:** image/video/audio/TTS/System One usage carries the debug span ([41650db](https://github.com/nurf-ai/ai/commit/41650db4495c7bacbd8ba86324b2706b8cbac341))
* **systemone:** keep zero noul/score/confidence in answer JSON ([41650db](https://github.com/nurf-ai/ai/commit/41650db4495c7bacbd8ba86324b2706b8cbac341))

## [0.11.1](https://github.com/nurf-ai/ai/compare/v0.11.0...v0.11.1) (2026-09-27)


### Bug Fixes

* **anthropic:** omit the system block when the prompt is empty ([932fcbe](https://github.com/nurf-ai/ai/commit/932fcbe48164e1f07b364b1a7b4b437171b808dd))

## [0.11.0](https://github.com/nurf-ai/ai/compare/v0.10.0...v0.11.0) (2026-09-20)


### ⚠ BREAKING CHANGES

* JudgmentProvider renamed to SystemOneProvider, all related types and functions renamed accordingly. The JavaScript-facing names changed accordingly.

### Features

* rename JudgmentProvider → SystemOneProvider (Typesafe System One) ([f818e41](https://github.com/nurf-ai/ai/commit/f818e411e65e62ecd01bc055ec1f7b6521259c72))

## [0.10.0](https://github.com/nurf-ai/ai/compare/v0.9.3...v0.10.0) (2026-09-19)


### Features

* add SystemOneProvider interface with Typesafe (Jev) implementation ([92ba580](https://github.com/nurf-ai/ai/commit/92ba5800233472bfd4061bda34249df408962d6c))
* add SystemOneProvider with Typesafe (Jev) implementation ([bb76e3c](https://github.com/nurf-ai/ai/commit/bb76e3c77e9a98372208d6bacf423193985ab2ee))

## [0.9.3](https://github.com/nurf-ai/ai/compare/v0.9.2...v0.9.3) (2026-09-09)


### Bug Fixes

* handle GA realtime event names, surface caller transcript ([c4a147f](https://github.com/nurf-ai/ai/commit/c4a147f52e3b9c9291c2665bd5a010bf0d0a52ea))

## [0.9.2](https://github.com/nurf-ai/ai/compare/v0.9.1...v0.9.2) (2026-09-09)


### Bug Fixes

* realtime GA audio format wire shape ([c71dc6f](https://github.com/nurf-ai/ai/commit/c71dc6f85768270c28056d9a5d6b39b9807b49ee))
* realtime GA audio format wire shape ([c02dfa0](https://github.com/nurf-ai/ai/commit/c02dfa07a9d7e0fec502a618e97716c9596179f4))

## [0.9.1](https://github.com/nurf-ai/ai/compare/v0.9.0...v0.9.1) (2026-09-09)


### Bug Fixes

* remove Temperature from RealtimeSessionConfig ([974ac60](https://github.com/nurf-ai/ai/commit/974ac608157604853c81a3490af9384fbf5d7607))

## [0.9.0](https://github.com/nurf-ai/ai/compare/v0.8.0...v0.9.0) (2026-09-09)


### Features

* add AudioProvider interface + fal Sonilo TTSFX/TTMusic ([6015df9](https://github.com/nurf-ai/ai/commit/6015df9c150c5f2243c0859597192b03bb5e6f80))
* add AudioProvider interface + fal Sonilo TTSFX/TTMusic ([4014f37](https://github.com/nurf-ai/ai/commit/4014f3711ad6d2a172b5e7df7b9e192acac0afc6))

## [0.8.0](https://github.com/nurf-ai/ai/compare/v0.7.2...v0.8.0) (2026-09-08)


### Features

* add RealtimeProvider for OpenAI Realtime API ([a837650](https://github.com/nurf-ai/ai/commit/a8376508a2005304037101dc8201128be5488832))
* add RealtimeProvider for OpenAI Realtime API (WebSocket voice) ([2fcd50e](https://github.com/nurf-ai/ai/commit/2fcd50e44b9508bce3f6caa7c2ad7110f3128e5f))


### Bug Fixes

* correct gpt-realtime-2 pricing ($4/$24/$32/$64) ([40134b6](https://github.com/nurf-ai/ai/commit/40134b6769d5c506ea651c629f06ee2d400555a2))
* realtime GA API format, integration tests, matrix update ([1e20b27](https://github.com/nurf-ai/ai/commit/1e20b27d40b9bb99ed9f6083a9aa8167b765ba8b))
* restore known-good matrix rows, reorder veo after gemini-omni ([9ec1881](https://github.com/nurf-ai/ai/commit/9ec188194470f205664bc0d91da80a95d961b6f6))

## [0.7.2](https://github.com/nurf-ai/ai/compare/v0.7.1...v0.7.2) (2026-09-08)


### Bug Fixes

* **errors:** a spent account is not a rate limit, and viewers are told neither ([eddc5fd](https://github.com/nurf-ai/ai/commit/eddc5fdaf07165f3cc6a7cc4a4a56d9e0fc0e572))

## [0.7.1](https://github.com/nurf-ai/ai/compare/v0.7.0...v0.7.1) (2026-09-06)


### Bug Fixes

* change default TTS voice to Coral ([edb636f](https://github.com/nurf-ai/ai/commit/edb636f6141be8468afc24d032b565c384819bc0))

## [0.7.0](https://github.com/nurf-ai/ai/compare/v0.6.0...v0.7.0) (2026-09-06)


### Features

* text-to-speech provider (OpenAI gpt-4o-mini-tts) ([f207ffe](https://github.com/nurf-ai/ai/commit/f207ffe62188c6f4f83488f0e2de87da07a8e03c))

## [0.6.0](https://github.com/nurf-ai/ai/compare/v0.5.3...v0.6.0) (2026-09-05)


### Features

* **stream:** emit tool-call argument deltas in StreamChunk ([4f2acdf](https://github.com/nurf-ai/ai/commit/4f2acdf38a9a11081ffa748ce183da907f90a41c))

## [0.5.3](https://github.com/nurf-ai/ai/compare/v0.5.2...v0.5.3) (2026-09-03)


### Bug Fixes

* **openai:** retry function tools with reasoning_effort none when the model refuses reasoning + tools ([d47b89c](https://github.com/nurf-ai/ai/commit/d47b89c4d2af0e9156901fe3e4774ac88e47e0cf))

## [0.5.2](https://github.com/nurf-ai/ai/compare/v0.5.1...v0.5.2) (2026-09-03)


### Bug Fixes

* **openai:** skip empty system prompt in structured output calls ([f45d910](https://github.com/nurf-ai/ai/commit/f45d910edd00c25ddc81b5905c189500fc198369))

## [0.5.1](https://github.com/nurf-ai/ai/compare/v0.5.0...v0.5.1) (2026-09-03)


### Bug Fixes

* **minimax:** map requested resolution onto H3's tiers (480P/768P/2K) ([d030234](https://github.com/nurf-ai/ai/commit/d0302347f01854cff0cf02223e1433a5f1919273))
* **veo:** fit resolution/duration to Veo 3.1 limits (1080p only at 8 s) ([a4fd7fd](https://github.com/nurf-ai/ai/commit/a4fd7fdbd914166b5210d54f89b5c211e261ce05))
* **video:** return clip bytes for keyed urls, log router fallbacks, MiniMax resolution case ([9ce7693](https://github.com/nurf-ai/ai/commit/9ce7693c6c85bec2c07da6710e78bfd2f0dd76d7))

## [0.5.0](https://github.com/nurf-ai/ai/compare/v0.4.2...v0.5.0) (2026-09-03)


### Features

* **video:** route to the requested model's provider first, fall back on defaults ([61cf34f](https://github.com/nurf-ai/ai/commit/61cf34f5789b50c34a0524e8bd47a987ef686c8a))

## [0.4.2](https://github.com/nurf-ai/ai/compare/v0.4.1...v0.4.2) (2026-09-02)


### Bug Fixes

* move Veo under Gemini provider in testmatrix ([48571cd](https://github.com/nurf-ai/ai/commit/48571cd62198c23fd63e13a14665211674342f5f))

## [0.4.1](https://github.com/nurf-ai/ai/compare/v0.4.0...v0.4.1) (2026-09-02)


### Bug Fixes

* penalize unpriced models in video router price scoring ([fe96bb8](https://github.com/nurf-ai/ai/commit/fe96bb846120b1042b5c7e130cb272f9ad1285f7))

## [0.4.0](https://github.com/nurf-ai/ai/compare/v0.3.0...v0.4.0) (2026-09-02)


### Features

* add Gemini video provider (Omni Flash Interactions API) ([4fa1c98](https://github.com/nurf-ai/ai/commit/4fa1c98b44c6f182b2e549405857b5013b47551c))
* add Veo 3.1 and MiniMax H3 direct video providers ([74cb5cf](https://github.com/nurf-ai/ai/commit/74cb5cfaa0f3463c471d4d7a79349fb104bf8d0f))
* add VideoRouter with generic multi-dimension routing ([da1a3de](https://github.com/nurf-ai/ai/commit/da1a3de995170965b22da75b40cfa3cb7830b990))
* per-token pricing for video/image, update model catalog ([f533045](https://github.com/nurf-ai/ai/commit/f533045faa7158dd70f1c4f46fb14f390a04214c))


### Bug Fixes

* gemini video duration via prompt timestamps, default 10s, text_to_video task ([121e3ba](https://github.com/nurf-ai/ai/commit/121e3bacb223b53363dd899dc12489986c8641a6))
* veo provider durationSeconds type, response envelope, I2V url-only ([b81c71b](https://github.com/nurf-ai/ai/commit/b81c71bd8ca065d7ba9b237a747cba658eb3f21c))

## [0.3.0](https://github.com/nurf-ai/ai/compare/v0.2.0...v0.3.0) (2026-09-02)


### Features

* context-stamped meter metadata, fal LTX-0.9 + auto t2v endpoint ([7519516](https://github.com/nurf-ai/ai/commit/75195164392e180f37e264b66cdf034c3341674b))

## [0.2.0](https://github.com/nurf-ai/ai/compare/v0.1.0...v0.2.0) (2026-08-31)


### Features

* add fal.ai video provider, VideoProvider interface, multimodal parts ([333a954](https://github.com/nurf-ai/ai/commit/333a954d18008536da52ea4bb434d360edcd1466))


### Bug Fixes

* openai gpt-image-1 edit, add image edit/caching tests ([9d67d00](https://github.com/nurf-ai/ai/commit/9d67d00529afa0c154367fb7d4af606b12535cd9))
