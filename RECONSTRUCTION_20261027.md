# gsplat reconstruction release — 2026-10-27

The default reconstruction implementation now uses gsplat 1.5.3 (Apache-2.0) and pycolmap 4.2.1, replacing the GraphDeco executable and its extension installation. Upstream attribution and license are in LICENSE.gsplat. This does not grant rights to gated third-party models.

## Reproducible deployment

Run `python package_gsplat.py` to produce a deterministic source ZIP, SHA-256 and Runpod deployment configuration. The bundle contains source and an upstream license only, never credentials, weights or user images. Bootstrap checks the exact inventory and digest before installing pinned dependencies and starting the worker. The immutable base image is `runpod/flash@sha256:039efd6638ec6d6728246bb5130d8976c2f39be357c4ddd1e71866b238474848`.

The running source release SHA-256 is `4cc92ebbfefb8b6b8d77bb5f04bb492fe28b668a2e5f308b10159b6ffc088d82`. Endpoint `0v3b5pbqye54nc` is min 0/max 1, idle 5 seconds, AMPERE_24 and a 600-second job timeout. CUDA kernels compile on the worker's first training invocation. The checked GPU catalog rate was $0.69 per active worker-hour; this is not an invoice or an account spending cap. The earlier short-test source was `9d1629fe8af03eee8eb8062f4179a56a70dc6852a2608ab108bc732f3d758556`.

The Dockerfile offers a preinstalled alternative and preserves the Node/splat-transform toolchain. The source-bundle runtime was exercised live. GHCR build run 40580344426 succeeded for merge 5e7c97ba9fa9f4189595cb81ffc6d706399dfd82; it is a separately built artifact and is not the source-bundle deployment.

## Actual tests

- Image-only fixture job `a8fcb679-a387-462a-99ad-72d89e07824c-u2` recovered cameras for all 12 original raytraced images, with 4,890 sparse points. No known camera poses were supplied to COLMAP.
- Actual GPU training ran 120 iterations, yielding 4,890 Gaussians and a 1,155,515-byte PLY. SHA-256: `f3ac2c34277f6f3a2df70f4e85fcc8008517da41c700fb3e172fa49e84fdfe5e`. Loss 0.418718 → 0.390598; Runpod execution 91,969 ms.
- A separate known-pose sphere job trained 400 Gaussians, loss 0.249067 → 0.112313. It verifies training, not camera recovery.
- PlayCanvas loaded both resulting PLY files and rendered frames without JavaScript errors. The image-only fixture's short-training result is visually poor; this is explicitly NOT property-quality acceptance.
- 25 offline Python tests passed for transfer restrictions, private protocol, source provenance, packaging and operator actions.

These fixtures are synthetic, not customer scans or physical scanner tests. Coordinates are SfM-relative and are not metric measurements.

## Density budget correction

A 3,000-iteration operator diagnostic on source `c8c6cc812b8b0622303b8ed9b5c106bacc3d4dbaa3e888a5c1e71116f5216733` failed after 124,306 ms with `gaussian_count_budget_exceeded` (job `64104e64-7a4f-40e5-8717-78381bedb624-u1`). DefaultStrategy could grow beyond the 200,000-Gaussian budget before the next iteration checked it.

The corrected implementation stops densification before a clone/split pass can exceed the budget, while continuing optimization of the existing population. Screen-space splitting is disabled in the pinned strategy; clone and split masks are disjoint, so a pass can at most double the count. The hard count check and the 300-second training deadline remain in place. Statistics report the density-freeze iteration, maximum count and last training-view PSNR. This PSNR is not a held-out accuracy metric.

The exact `photogrammetry_quality` operator action uses the owned image-only fixture and 3,000 fixed iterations; client-supplied options are rejected. It returns a training-view PNG and includes the full PLY only when it is at most 8 MB. Larger geometry stays omitted from this diagnostic response, rather than exceeding provider response limits. No customer processing is enabled by this action.

Fixed-source GPU regression `f1129699-78ea-4a4e-a1ba-60d87d627dfb-u1` completed all 3,000 iterations on worker `myn51716on32b2` in 139,207 ms. Densification froze at iteration 1,101; final population was 134,998, below the unchanged 200,000 limit. Loss improved from 0.406911 to 0.111359. The worker reported a 31,861,005-byte PLY and SHA-256 `f3f2b55f9a3abce7b1b9a15c77f7c1711af0039005be10edb2e4dfbcc1005f94`; the full file was omitted by the diagnostic response cap, so these are worker-reported metadata, not independent full-file verification. The returned training preview shows a recognizable textured cube with floor/background floaters. Last training-view PSNR was 15.35 dB; this is neither held-out validation nor acceptance for real-property quality. See `density-budget-proof-20261027.json`.

An earlier fixed-source attempt (`7da2e616-64ea-4db9-b365-b00e83751d04-u2`) was no longer available from Runpod when the chat resumed. Its outcome is explicitly unverified. The recorded regression above was a new bounded diagnostic after checking that no job remained running. No provider charge is inferred from missing results or elapsed time.

## Private protocol and activation

Only exact operator diagnostic actions and the existing `arqary-private-scan-v1` protocol are routed. Arbitrary public legacy jobs are rejected. Private jobs remain disabled by default (`ARQARY_PRIVATE_PROTOCOL_ENABLED`); enabling them also needs scoped input/output host allowlists and the existing manifest, identity and byte-integrity checks. Do not pass arbitrary hosts or credentials in job payloads.

The platform's customer reconstruction submission remains disabled. It still needs legitimate signed-in acceptance, authenticated provider cost evidence, private storage transfer acceptance, and quality/timing validation on representative real captures. The 3,000-iteration default has a bounded 300-second training deadline and has not been quality-benchmarked. Timeout and Gaussian-count breaches fail explicitly rather than exporting a partial success.

`facebook/VGGT-1B-Commercial` remains manually gated. This engine supplies camera recovery through COLMAP; it does not claim to run VGGT or bypass its license/access requirements.

