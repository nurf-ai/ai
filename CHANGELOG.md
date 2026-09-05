# Changelog

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
