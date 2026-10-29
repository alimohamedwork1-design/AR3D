# Spirula: operator-only reconstruction experiment

This is an opt-in CLI alongside the existing gsplat worker, not a replacement
for it and not a customer dispatch endpoint. No production endpoint, customer
execution flag, financial admission rule or Basira quota is changed.

The adapter invokes an independently installed Spirula executable. It does not
bundle upstream source or binaries. Spirula is GPL-3.0; the existing gsplat
backend remains Apache-2.0. Review distribution/dependency obligations before
shipping a binary or offering a broader integration.

## Pinned installation

- Upstream: https://github.com/harry7557558/spirula-studio
- Release: `v2026.10.16`, commit `65246c5579c127081bff6f4422bd12a542906f78`
- Linux archive: https://github.com/harry7557558/spirula-studio/releases/download/v2026.10.16/spirula-2026.10.16-ubuntu-vulkan-x86_64.zip
- Archive SHA-256: `df5bb7acbfea63c16497c229b70c7a05ed13c22807554dee83cd6415aa411bf3`
- Extracted executable SHA-256: `f29ef66ed881a3d5a5c9f2e1f1bb587c21617442eed16ee9d11830150f41ae03`

Use a GPU container exposing NVIDIA graphics capabilities, not CUDA alone.
On the verified Ubuntu 24.04 Runpod image the missing `libegl1` caused Vulkan's
ICD to return no instance function despite CUDA working. Installing the EGL/GLX
loader dependencies resolved it; do not replace the host kernel driver.
The verified image resolved to
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.

```bash
apt-get update
apt-get install -y libvulkan1 vulkan-tools libopengl0 libegl1 libglx0 libxrandr2 libgomp1 libglib2.0-0 ffmpeg
python -m pip install pillow==11.3.0 numpy plyfile==1.1.3 pycolmap==4.2.1
vulkaninfo --summary
python experimental/spirula_backend.py --experimental \
  --binary /opt/spirula/spirula --images /data/owned-capture/images \
  --output /data/new-experiment --iterations 3000 --timeout 600
```

Download/unzip the pinned archive and verify both hashes before the invocation.
The adapter checks the executable digest again. No installer executes from an
unversioned URL. Camera recovery/training are native Vulkan; `pycolmap` reads and
validates its model. This experiment accepts 10–200 local PNG/JPEG images,
256 MiB total and 24 MP per input, prepared at a maximum 640 pixels per side.
It does not yet accept direct video or certify 360 capture, meshing or metric scale.

## Output and safeguards

- New output directory only; no overwriting earlier evidence.
- Shared 1–900 second deadline includes preparation, camera recovery and training.
  Timed-out native process groups are killed; partial results are never successful.
- Explicitly bounded training: 100–5,000 iterations, up to 200,000 Gaussians.
- At least 80% registered views and at least ten; sparse seeds must fit the cap.
- Exact final checkpoint required, finite Gaussian attributes/valid rotations,
  at most 128 MiB; inference-only quaternion normalization preserves orientation.
- Three-dimensional coordinates remain relative; no measurement accuracy claim.
- Real held-out metrics and a captured preview camera are retained. A generic
  bounds fit can show unseen sides of a partial capture; use `preview_camera`
  when preparing a PlayCanvas viewer. Its coordinates account for the viewer's
  180-degree Z rotation and Spirula's scene transform.
- `point_cloud.ply` has byte count/digest in `experiment.json` and can be converted
  through the existing PlayCanvas SplatTransform path. The manifest always says
  `eligible_for_publication: false`. This is a local candidate, not a settled,
  owner-bound private platform artifact. Existing settlement/publication rules
  still apply; never directly insert it into customer inventory.

Acceptance and quality limits: [SPIRULA_EVALUATION_20261029.md](../SPIRULA_EVALUATION_20261029.md).
