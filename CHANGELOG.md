# Changelog

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
