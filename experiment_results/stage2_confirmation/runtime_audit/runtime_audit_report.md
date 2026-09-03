# Stage 2 Runtime Audit — E0 vs E1

Read-only audit. No training, resume, re-validation, or checkpoint/result modification was
performed. All data below comes from existing `training.log`, `training_epochs.csv`,
`metrics.csv`, `runtime.json`, `config_snapshot.yaml`, and checkpoint metadata (loaded with
`torch.load`, not executed), plus static inspection of the training code. Raw outputs live in
this directory: `runtime_audit.json`, `epoch_runtime_comparison.csv`,
`validation_runtime_comparison.csv`, `config_diff.txt`.

## 1. Workload comparison
- Train images: E0 = 1856, E1 = 1856 (match)
- Batch size: 16 (both), `drop_last=False` (`lib/helpers/dataloader_helper.py`)
- Expected iterations/epoch: ceil(1856/16) = **116** (both) — no partial last batch (1856/16 is exact)
- No gradient accumulation anywhere in `train_one_epoch` — one `optimizer.step()` per batch
- Total iterations/optimizer steps over 40 epochs: **4640** (both, identical)
- Both checkpoints report `epoch: 40` — both ran the full schedule, no early exit

## 2. DataLoader comparison
Both built via the same `build_dataloader(cfg['dataset'])` call (`workers=4` default,
`shuffle=True` train / `False` val, `pin_memory=False`, no custom sampler,
`persistent_workers`/`prefetch_factor` not set — PyTorch defaults for both). `train_split:
train_confirmation_50` in both `config_snapshot.yaml`, both logged 1856 images at startup.
Split file is a single shared file (`data/KITTIDataset/ImageSets/train_confirmation_50.txt`,
generated once in the previous session, sha256 `a2a6cd...` — not regenerated).

## 3. Config snapshot diff
Only one difference between `E0_baseline/config_snapshot.yaml` and
`E1_soft_ncf/config_snapshot.yaml` across the entire flattened config tree:
```
model.pseudo_depth_mode: center  vs  soft_ncf
```
No other key differs — batch size, workers, optimizer, LR, weight decay, scheduler,
milestones, backbone, pretrained flag, input size, augmentation, queries, depth bins, splits,
and seed are byte-identical.

## 4. Initialization / resume
Neither run used `--resume` (command lines had no such flag; no "Resumed from" log line in
either `training.log`). Both started at `start_epoch = 1` with a fresh pretrained-backbone
model. `experiment_initialization` block in both snapshots shows identical seeds
(444/197136/87528384/38862602496) and `full_model_checkpoint_initialization: false`.

## 5. Model / parameter comparison
Loaded `checkpoints/latest.pth` for both (metadata only, no forward pass):
- `model_state` key count: 582 (both) — key **sets are identical**, no shape mismatches
- optimizer: 2 param groups (141 + 199 tensors) — identical in both
- Final LR: 2e-6 in both (matches two 0.1× decays at milestones 25 and 33)

Architecture and trainable-parameter count are confirmed identical.

## 6. Forward / loss computational path
`monodetr.py:loss_depth_map` calls `self.ddn_loss(...)` with `mode=self.pseudo_depth_mode`;
everything else (backbone, encoder/decoder, detection heads) is untouched by this branch.
Inside `ddn_loss.py:forward`, `'center'` builds a hard per-pixel class-index target
(`build_target_depth_from_3dcenter` → `bin_depths`) and calls `FocalLoss` with an int64
target (which internally does an extra `one_hot` `scatter_`); `'soft_ncf'` builds a dense
float target directly (`build_soft_ncf_target`) and calls the same `FocalLoss`, which routes
to `soft_focal_loss` (no `scatter_`, direct weighted sum). Both target-builders use comparable
per-object Python loops; both loss variants operate on the same (B, 81, 24, 80) tensor. Given
that this loss sits on top of a ResNet-50 backbone + transformer encoder/decoder forward and
backward pass over 1280×384 images, this is a negligible fraction of total per-iteration
compute — it cannot plausibly account for a sustained 40% wall-clock gap. No branch is skipped,
detached, or omitted for either mode; no evidence of E1 skipping detection losses.

## 7. Loss progression sanity (workload validity only)
| Epoch | E0 train_loss_total | E1 train_loss_total |
|---|---|---|
| 1 | 32.963 | 32.988 |
| 10 | 13.228 | 13.423 |
| 20 | 10.123 | 10.141 |
| 30 | 7.038 | 7.084 |
| 40 | 6.530 | 6.647 |

Both finite throughout, near-identical magnitude and decay shape at every checkpoint. No sign
of E1 omitting loss terms to "cheat" a lower wall-clock cost.

## 8. AMP / backend / compile
- `grep` across `tools/run_pseudo_depth_experiment.py`, `lib/helpers/`, `monodetr.py`: **no**
  `autocast`, `GradScaler`, `torch.amp`, or `torch.cuda.amp` anywhere in the pipeline.
- **No** `torch.compile` / dynamo / inductor / cudagraph usage anywhere.
- `cudnn.benchmark = False`, `cudnn.deterministic = True` — set identically by
  `set_random_seed(444)`, called once per run before `build_dataloader`, same for both.
- No TF32 / `matmul_precision` configuration found (framework defaults apply to both equally).

Both runs are plain FP32, non-compiled, non-autotuned. This rules out AMP/precision/compile
differences as a cause.

## 9. Epoch-by-epoch runtime analysis (training)
| | E0 | E1 |
|---|---|---|
| mean | 1161.70s | 700.82s |
| median | 1159.38s | 700.13s |
| std | 5.90s (0.5%) | 2.99s (0.4%) |
| min / max | 1154.82 / 1181.16 | 698.23 / 717.18 |
| first-5 mean | 1168.41s | 700.40s |
| mid-5 mean | 1164.18s | 701.05s |
| last-5 mean | 1157.49s | 699.28s |

Both runs are **flat and stable from epoch 1 to epoch 40** (std < 1% of mean for both). There
is no ramp-up, no ramp-down, no discontinuity at the LR milestones (epoch 25: E0=1159.4s,
E1=700.2s; epoch 33: E0=1158.7s, E1=717.2s — E1's epoch 33 is its single-run maximum, +2.3%
above its own mean, but this is a one-epoch blip, not a sustained shift, and E0 shows no
corresponding bump at either milestone). **The 40% gap is present from epoch 1 and persists
identically through epoch 40 in both runs** — this rules out compile/cache warmup (which would
front-load in one run only) and rules out progressive thermal degradation within a run (which
would show a ramp, not a flat offset).

## 10. Validation runtime analysis
| | E0 | E1 | diff |
|---|---|---|---|
| inference mean | 624.11s | 337.34s | **-45.9%** |
| evaluator mean | 4.18s | 4.27s | +2.1% (noise) |

The CPU-bound official-evaluator portion (numba `rotate_iou` + KITTI matching) is
**statistically indistinguishable** between the two runs. Only the GPU-bound `helper.inference()`
portion (model forward pass over 3769 validation images) shows the large gap — same
magnitude and direction as the training-time gap. This localizes the anomaly specifically to
GPU-bound forward compute, not to the dataloader, CPU, or evaluator code path.

## 11. Detection-count / evaluator workload
Final validation snapshot (epoch 40 for both, `outputs/data/`, not re-run): 3769 prediction
files each; E0 = 17488 total detection lines (4.64/image), E1 = 17659 (4.69/image) — within
~1% of each other. Evaluator workload is essentially identical, consistent with §10's finding
that evaluator time itself did not differ.

## 12. GPU memory metric meaning
`peak_gpu_memory_MB` = `torch.cuda.max_memory_allocated() / 1024**2`, reset once via
`torch.cuda.reset_peak_memory_stats()` immediately before the epoch loop
(`tools/run_pseudo_depth_experiment.py`). This is PyTorch's own count of the peak bytes its
caching allocator actually handed out to tensors — **not** `max_memory_reserved()`, not an
NVML/Task-Manager reading, and not an OS-level "shared GPU memory" figure. Both runs report
~20458 MB via this exact same code path (E0 20458.7, E1 20458.1 — 0.7 MB apart, effectively
identical). No CUDA OOM error appears in either `training.log`. Given the reported physical
VRAM is 16380 MiB (from an `nvidia-smi` check earlier in this session, not logged inside the
training runs themselves), an allocator peak above that with no OOM is most plausibly explained
by the Windows driver's shared/system-memory oversubscription for CUDA contexts — but **this
session's logs contain no NVML/driver telemetry to confirm that mechanism directly**, so it is
reported here as the most consistent available explanation, not a confirmed measurement.

## 13. GPU/system telemetry availability
GPU utilization, temperature, clock, power draw, shared-GPU-memory, CPU utilization, system
RAM, and disk usage are **not present** in any existing log or `runtime.json` for either run.
Only `peak_gpu_memory_MB` (§12) exists. No new measurement was taken for this audit.

## 14. Runtime-difference cause classification
**SYSTEM/ENVIRONMENT LIKELY.**
- Code and config are identical except the intended `pseudo_depth_mode` (§3, §6, §8)
- Workload is identical: same images, same iteration/optimizer-step count, same architecture/
  parameter count (§1, §5)
- The extra compute inside the `'center'` loss path (§6) is far too small to explain a 40%
  gap on a ResNet-50+transformer model
- The gap is flat and present from epoch 1 in both runs (§9), and shows up almost identically
  in the GPU-bound validation-inference time while the CPU-bound evaluator time does not move
  at all (§10) — pointing specifically at GPU throughput during E0's execution window versus
  E1's, not at anything in this codebase.
- No log-based telemetry (§13) is available to name the exact external mechanism (thermal
  state, driver/OS power management, shared-memory fallback, or another factor), so the
  specific cause is not confirmed — only that it is external to the code/config/workload
  audited here.

## 15. Stage 2 experiment validity
**VALID WITH CAVEAT.**
- Same dataset (1856 images, identical split file), same iteration/optimizer-step counts
  (4640 each), same initialization policy (fresh pretrained backbone, seed 444, no resume),
  same architecture/parameter count (582 keys, identical shapes), same optimizer/scheduler
  (AdamW, milestones [25,33], decay 0.1, identical final LR), same AMP/precision/backend
  settings (none/FP32/cudnn deterministic), same evaluator workload (§11) — only the intended
  `pseudo_depth_mode` differs.
- The AP/depth-metric comparison itself is unaffected by wall-clock runtime — official KITTI
  AP is computed from saved model outputs against ground truth, independent of how long
  training took.
- The caveat: the runtime figures reported in the Stage 2 report (§13 there) should **not** be
  read as "soft_ncf is 40% cheaper than center" — that comparison is confounded by whatever
  external factor separated the two ~14h and ~9h execution windows. Future runtime comparisons
  between modes should ideally be run interleaved or on a freshly-reset system to avoid this
  confound.

## 16. Suggested next checks (no training required)
- If the system is later free, capture `nvidia-smi -l` (utilization/clock/temp) during a short
  (few-iteration) sanity run of each mode back-to-back on the same day, to see whether the gap
  reproduces without the ~14h/~9h separation — this would directly test the "environment
  changed between windows" hypothesis without touching the existing Stage 2 results.
- Check Windows Event Viewer / power-plan logs for the E0 execution window (2026-09-02 10:08
  to 2026-09-03 00:29) for any throttling, driver reset, or power-state change events, if such
  logs are still retained.
