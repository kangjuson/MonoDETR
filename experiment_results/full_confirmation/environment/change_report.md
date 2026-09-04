# Full-confirmation environment audit — change report

Windows local run → new Linux GPU server. Full audit performed before any full training was
started. No training, resume, or result/checkpoint modification was performed as part of this
audit; only environment setup (package install, one build-config fix, one dataset symlink) and
read-only verification.

## Change table

| Change ID | Category | Previous | Current | Severity | Action |
|---|---|---|---|---|---|
| CHG-001 | ENVIRONMENT | Python 3.13.14 (Windows) | Python 3.8.10 (Linux `.venv`) | MEDIUM | No action; verified compatible |
| CHG-002 | ENVIRONMENT | PyTorch 2.7.1+cu126 (Windows) | PyTorch 2.0.1+cu118, torchvision 0.15.2+cu118 (Linux) | HIGH | Installed per user's explicit choice; verified via smoke tests |
| CHG-003 | ENVIRONMENT | 1x RTX 4060 Ti 16GB, driver 591.44, cc 8.9 | 4x RTX A5000 24GB, driver 550.54.15, cc 8.6 | HIGH | Single GPU selected for the run (matches prior single-GPU condition) |
| CHG-004 | ENVIRONMENT | Numba 0.66.0 (paired with Python 3.13.14) | Numba 0.58.1 (paired with Python 3.8.10) | LOW | Re-ran rotate_iou synthetic test; results match |
| CHG-005 | DATASET | Windows local path | `/public/data/KITTIDataset` via new `data -> /public/data` symlink | LOW | Symlink only, no move/copy; counts verified exact |
| CHG-006 | EVALUATOR | `min()`→conditional-expression patch in rotate_iou.py CUDA kernel | Unchanged (present since first commit) | NONE | Re-verified via synthetic IoU test + evaluator import smoke |
| CHG-007 | BUILD/EXTENSION | No compiled extension existed; source gencode covered only sm_60/61/70/75 | Added sm_80/sm_86 (+ PTX) gencode; built with CUDA 11.8; no kernel source changed | MEDIUM | Built and smoke-tested (forward vs. PyTorch reference + gradcheck) |
| CHG-008 | ENVIRONMENT | `.venv` had only pip/setuptools | Full stack installed (see pip_freeze.txt) | HIGH | First-time install, not an upgrade/downgrade; verified via full test suite |
| CHG-009 | CONFIG | N/A (Stage1/2 used interval 5 over 25/40 epochs) | Launcher hardcodes validation interval = 5 for the full 195-epoch run too | LOW | No code change; cost is ~44 min vs. ~23 min for every-10, identical for both arms — flagged for sign-off, not changed unilaterally |

## Summary by impact

**SAFE CHANGES** (no effect on experiment meaning):
- CHG-005 (dataset path, symlink only — data itself byte-identical, counts verified)
- CHG-006 (evaluator patch unchanged — re-verified, not modified)

**CONTROLLED CHANGES** (changed, but applied identically to E0 and E1, so internal fairness is preserved):
- CHG-001 Python version
- CHG-002 PyTorch/CUDA stack
- CHG-003 GPU model (assuming single-GPU is used for both E0 and E1 runs)
- CHG-004 Numba version
- CHG-007 custom extension gencode fix
- CHG-008 full package install
- CHG-009 validation interval (pending explicit sign-off; identical for both arms regardless of which value is used)

**BLOCKING CHANGES** requiring resolution before training: **none found.** All checks below
(dataset, E0/E1/E2 implementation, custom op, evaluator, unit tests, one-batch smoke) passed.

## Note on cross-environment comparability

CHG-002 and CHG-003 together mean the *absolute* training-time and possibly some
convergence-sensitivity numbers from this run are not naturally comparable to a hypothetical
same-hardware Windows re-run — but Stage 1 and Stage 2 were both already run entirely on the
old Windows/4060 Ti/PyTorch-2.7.1 environment, and this full run is entirely on the new
Linux/A5000/PyTorch-2.0.1 environment. Every comparison that matters for the actual research
question (E0 vs E1 within this full run) has both arms under the exact same environment, code,
and config except `pseudo_depth_mode` — see the internal-fairness verdict in the main report.
