# Stage 1/2 Preprocessing & Training Logic Refinement Plan

## Background & Motivation
The current UI action prediction pipeline has four specific logical flaws in data preprocessing and training dataset construction. These flaws affect the robustness of Stage 1 (Template mapping), Stage 2 (Rule-based Fallback), and the quality of the LLM training set for Stage 3. This plan details the surgical fixes required to address these issues without altering the overall pipeline architecture.

## Scope & Impact
Modifications are strictly limited to two files:
- `src/preprocess.py`: Fix value extraction logic, improve keyword matching robustness, and enforce a majority-rule approach for site templates.
- `src/train.py`: Correct the data split ratio to provide more data for LLM training.
- No changes will be made to the overall inference flow or model architecture.

## Proposed Solution & Implementation Steps

### 1. `src/preprocess.py` Updates
**A. Fix `extract_value_from_task` (SELECT vs. Quoted String Priority):**
Currently, quoted strings are given priority over SELECT options, leading to incorrect value extraction if the task mentions a field name in quotes instead of the target option.
- **Change:** For `SELECT` operations, parse the `attrs` string for available options (`options=`) and check if any option exists within the task string **before** falling back to extracting quoted text.

**B. Fix `fallback_rule_based` (Substring False Positives):**
The simple `in` operator (e.g., `"enter " in task`) falsely matches substrings in unrelated words (e.g., "c**enter**").
- **Change:** Replace simple substring checks with word-boundary regular expressions (`\btype\b|\benter\b|\binput\b`) to ensure accurate operation classification.

**C. Fix `build_site_templates` & `predict_by_template` (Majority Rule for Templates):**
Storing all historical interactions for a `(site_token, step_num)` and returning the first match allows minority edge cases to override the standard flow.
- **Change:** Utilize `collections.Counter` in `build_site_templates` to store only the most common `(op, label, tag)` tuple for each key. Update `predict_by_template` to check against this single, highest-confidence template rather than iterating through a list.

### 2. `src/train.py` Updates
**A. Correct `train_test_split` Ratio:**
The current implementation splits the data leaving 70% for template generation and only 30% for LLM training. The LLM requires a larger dataset to generalize unseen tasks effectively.
- **Change:** Adjust `train_test_split(df, test_size=0.3)` to `test_size=0.7`, allocating 70% of the training data to the LLM and 30% to template building.

## Verification & Testing
- Inspect `src/preprocess.py` to ensure `Counter` is imported correctly and the refactored functions behave according to the new logic.
- Inspect `src/train.py` to ensure `test_size=0.7` is correctly set.
- Ensure the overall pipeline (Stage 1 -> 2 -> 3) remains intact and no syntax errors are introduced.
