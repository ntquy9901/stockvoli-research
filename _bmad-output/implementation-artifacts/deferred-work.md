# Deferred Work from Adversarial Code Review

**Date:** 2026-06-13
**Source:** Comprehensive adversarial code review + Edge Case Hunter Review

---

## Deferred Goals

### Goal 2: Data Pipeline Overhaul (Depends on Goal 1)

**Issues to Fix:** 4-6

**Scope:**
- Fix silent data loss in processing (warning + continue → fail fast)
- Make temporal split configurable (remove hardcoded 80/20)
- Add proper data quality validation (toothless → blocking)
- Fix missing RV_20 column silent failure
- Fix empty dataset edge case
- Fix test data boundary calculation error

**Dependencies:** Requires Goal 1 (Critical Architecture Refactor) to be complete

**Estimated Effort:** 6-8 hours

**Acceptance Criteria:**
- Data processing fails explicitly when stocks have insufficient data
- Temporal split ratio is configurable via config file
- Data quality validation blocks invalid data from being used
- Missing RV_20 columns cause explicit errors, not silent skips
- Empty datasets fail fast with clear error messages
- Test data boundary calculations don't silently exclude stocks

---

### Goal 3: Training Pipeline Robustness (Depends on Goal 2)

**Issues to Fix:** 7-9 + edge cases

**Scope:**
- Fix memory leaks (multiple dataloader creation)
- Fix checkpoint race condition (early stopping vs periodic checkpoint)
- Implement resume capability (no current checkpoint resume logic)
- Fix validation loss division by zero risk
- Fix training history save failure (weak exception handling)
- Fix checkpoint overwrite race condition
- Fix early stopping with zero epochs
- Fix batch size exceeding dataset size
- Fix numerical stability with zero volatility

**Dependencies:** Requires Goal 2 (Data Pipeline Overhaul) to be complete

**Estimated Effort:** 8-10 hours

**Acceptance Criteria:**
- Training script explicitly cleans up resources
- Checkpoint saves happen before early stopping checks
- Training can resume from checkpoint without losing progress
- Validation handles edge cases (zero batches, division by zero)
- Checkpoints saved with atomic writes or backup strategy
- Training validates config parameters (batch_size, num_epochs > 0)
- Numerical stability checks for extreme volatility values

---

### Goal 4: Code Quality & Robustness (Additional Edge Cases)

**Issues to Fix:** Additional edge cases discovered during review

**Scope:**
- Fix missing config_path storage in TimesFMVN30Finetuner class
- Fix config file not found error handling
- Fix duplicate stock data loading (multiple files per stock)
- Fix prefetch_factor compatibility checks
- Add migration guide/documentation for dependency versions
- Update all config files (not just 2 mentioned in spec)

**Dependencies:** Can be parallelized with Goals 2-3

**Estimated Effort:** 4-6 hours

**Acceptance Criteria:**
- Class properly stores config_path as instance variable
- Config file errors provide clear, actionable error messages
- Data loading handles duplicate stock files correctly
- PyTorch version compatibility validated
- All config files consistent (no duplicate fields)
- Clear documentation of dependency requirements

---

**Status:** Deferred - Will be tackled after Goal 1 is complete
