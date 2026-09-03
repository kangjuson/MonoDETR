# KITTI evaluator compatibility diagnosis

## Environment

- Python 3.13.14
- PyTorch 2.7.1+cu126; bundled CUDA runtime 12.6
- Numba 0.66.0; separate `numba-cuda` package is not installed
- NVIDIA driver 591.44; driver supports CUDA 13.1
- CUDA toolkit / nvcc 12.8.61
- GPU: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9

MonoDETR's README demonstrates Python 3.8 and PyTorch 1.9.0+cu111. Its
`requirements.txt` lists `numba` without a version constraint.

## Root cause

Import eagerly compiles `rotate_iou_kernel_eval`, whose explicit signature is
six arguments and whose Python definition also has six arguments. The device
function `devRotateIoUEval` likewise has three signature/Python arguments.
Those declarations therefore match.

The misleading `Signature mismatch: 2 argument types given, function takes 1`
is emitted while Numba types the kernel's two calls to Python `min(a, b)`.
Under Python 3.13.14 and Numba 0.66.0 CUDA compilation, that built-in is
resolved as a single-argument callable. Replacing only the two `min` calls
with equivalent conditional expressions fixes eager JIT and execution.

No package was installed, removed, upgraded, or downgraded. No IoU arithmetic,
KITTI matching, difficulty definition, interpolation, R40 definition, sorting,
or recall threshold was changed.

## Verification

- Identical rotated boxes: IoU 1.0
- Disjoint rotated boxes: IoU 0.0
- Partial overlap: IoU 0.3333333433
- Official evaluator import: pass
- Full 3,769-image validation and official R40 evaluation: pass

See `environment.txt`, `import_smoke.json`, `rotate_iou_synthetic.json`, and
`../evaluator_smoke/smoke_status.json` for machine-readable results.
