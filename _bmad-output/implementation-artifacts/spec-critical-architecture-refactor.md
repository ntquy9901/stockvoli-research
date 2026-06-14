---
title: 'Critical Architecture Refactor'
type: 'refactor'
created: '2026-06-13'
status: 'done'
context: ['{project-root}/CLAUDE.md']
baseline_commit: 'b150703f421c402975b45ba90687fddc0c956255'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Training script uses dangerous monkey patching (lines 36-78) and global environment manipulation (line 67) to hide dependency incompatibilities, plus config has duplicate fields causing bugs. Recent KeyError proved duplicates cause real failures.

**Approach:** Remove all monkey patching, use compatible dependency versions, move environment setup to main(), and consolidate duplicate config fields to single canonical names.

## Boundaries & Constraints

**Always:**
- Remove ALL monkey patching code (lines 36-78 in model_training_fixed.py)
- Remove ALL global environment manipulation at import time
- Keep `num_epochs` and `save_every_n_epochs` as canonical field names
- Code must import cleanly without runtime patches
- Training must continue to work after refactor

**Ask First:**
- Should CUDA memory config (`expandable_segments:True`) be kept, made configurable, or removed entirely?
- If bitsandbytes causes import errors, should we: (A) upgrade PyTorch/transformers to compatible versions, or (B) accept we can't use current transformers version?

**Never:**
- Do NOT add new monkey patches or workarounds
- Do NOT break existing training functionality
- Do NOT change field names to `epochs` or `save_every` (code uses `num_epochs` and `save_every_n_epochs`)
- Do NOT leave any compatibility shims in place

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Import training script | Clean Python environment | Module imports without side effects or patches | N/A |
| Load config | config.yaml with old duplicate fields | Deprecation warning if using old field names, use canonical names | Error if neither old nor new fields present |
| Training with new code | Same data, same config | Training completes successfully, checkpoints save | N/A |
| Import without patches | Current PyTorch/transformers versions | Clean import OR clear error about incompatible versions | Fail-fast with actionable error message |

</frozen-after-approval>

## Code Map

- `src/model_training_fixed.py` -- Main training script with monkey patching (lines 36-78) and global env setup (line 67)
- `configs/config.yaml` -- Config with duplicate fields (lines 44-45: `epochs`/`num_epochs`, lines 80-81: `save_every`/`save_every_n_epochs`)
- `requirements.txt` -- Dependency versions that may need updating for compatibility
- `tests/test_colab_preflight.py` -- Existing pre-flight tests (may need update to verify clean imports)

## Tasks & Acceptance

**Execution:**
- [x] `src/model_training_fixed.py` -- Remove lines 36-38 (torch.float8_e8m0fnu workaround) -- Replaced with compatible dependency versions
- [x] `src/model_training_fixed.py` -- Remove lines 43-77 (all monkey patching code) -- Code now imports transformers/PEFT cleanly
- [x] `src/model_training_fixed.py` -- Move line 67 (os.environ CUDA config) into main() function -- Environment setup now explicit in main()
- [x] `configs/config.yaml` -- Remove `epochs: 100` (line 44) and `save_every: 10` (line 80) -- Only `num_epochs` and `save_every_n_epochs` remain
- [x] `configs/config_staging.yaml` -- Apply same duplicate field removals -- Consistent across all config files
- [x] `requirements.txt` -- Update dependency versions for compatibility -- transformers 4.x for PyTorch 2.5 compatibility
- [x] `tests/test_colab_preflight.py` -- Add test that verifies clean module import -- 3 new tests verify no monkey patching

**Acceptance Criteria:**
- Given clean Python environment, when importing model_training_fixed module, then it imports without side effects or patches
- Given config with old duplicate fields, when loading config, then deprecation warning issued but training uses canonical field names
- Given training script execution, when reaching main(), then CUDA memory config is set explicitly (not as import side effect)
- Given current PyTorch/transformers versions, when importing transformers/PEFT, then they import successfully OR clear error about incompatibility (no silent patches)
- Given refactored code, when running training, then training completes successfully with same results as before

## Spec Change Log

## Design Notes

**Field Name Investigation:**
- Code consistently uses `num_epochs` (lines 949, 952, 976, 1175, 1176)
- Code consistently uses `save_every_n_epochs` (line 1022)
- Therefore keep `num_epochs` and `save_every_n_epochs` as canonical names
- Remove `epochs` and `save_every` to eliminate confusion

**Dependency Compatibility Issues:**
- Lines 37-38: Workaround for missing `torch.float8_e8m0fnu` (PyTorch too old for transformers 4.35+)
- Lines 43-77: Monkey patching to hide bitsandbytes from PEFT (PEFT tries to load it but it's not installed)
- `requirements.txt` shows: transformers>=4.35.0, peft>=0.5.0, torch>=2.0.0
- Root cause: Version mismatch between PyTorch and transformers/PEFT

**CUDA Memory Config (line 67):**
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` globally at import time
- Comment claims this "prevents memory fragmentation on 8GB GPU"
- Proper fix: Move to main() so it's explicit and configurable, not hidden import side effect

## Verification

**Commands:**
- `python -c "import src.model_training_fixed; print('Import OK')" -- expected: Clean import without patches or warnings
- `python scripts/colab_preflight.py --verbose -- expected: All 6 checks pass, no warnings about config
- `pytest tests/test_colab_preflight.py -v --expected: All 43 tests pass (or updated count if test added)

**Manual checks:**
- Open `src/model_training_fixed.py`, verify lines 36-78 contain NO monkey patching code
- Open `configs/config.yaml`, verify NO duplicate epoch or save_every fields
- Run training script locally, verify it completes first epoch successfully

---

## Suggested Review Order

**Monkey Patching Removal (Core Refactor)**

- Critical: Removed torch.float8_e8m0fnu workaround that hid dependency incompatibility
  [`model_training_fixed.py:33`](../../src/model_training_fixed.py#L33)
- Critical: Removed 35 lines of monkey patching (importlib, bitsandbytes, PEFT utilities)
  [`model_training_fixed.py:36`](../../src/model_training_fixed.py#L36)
- Improvement: Added 3 tests to verify no monkey patching remains in code
  [`test_colab_preflight.py:199`](../../tests/test_colab_preflight.py#L199)

**Global Environment Manipulation Fix**

- Moved CUDA memory config from import-time to explicit main() setup
  [`model_training_fixed.py:1114`](../../src/model_training_fixed.py#L1114)
- Test verifies CUDA config not set during import, only in main()
  [`test_colab_preflight.py:245`](../../tests/test_colab_preflight.py#L245)

**Config Field Consolidation**

- Removed duplicate `epochs: 100` field (kept canonical `num_epochs: 100`)
  [`config.yaml:44`](../../configs/config.yaml#L44)
- Removed duplicate `save_every: 10` field (kept canonical `save_every_n_epochs: 10`)
  [`config.yaml:80`](../../configs/config.yaml#L80)
- Staging config updated identically for consistency
  [`config_staging.yaml:44`](../../configs/config_staging.yaml#L44)
  [`config_staging.yaml:77`](../../configs/config_staging.yaml#L77)

**Additional Config Files (Patch Fix)**

- Removed duplicate fields from 5 additional config files for consistency
  [`config_complete.yaml`](../../configs/config_complete.yaml)
  [`config_g4.yaml`](../../configs/config_g4.yaml)
  [`config_t4.yaml`](../../configs/config_t4.yaml)
  [`config_t4_optimized.yaml`](../../configs/config_t4_optimized.yaml)
  [`config_working.yaml`](../../configs/config_working.yaml)
