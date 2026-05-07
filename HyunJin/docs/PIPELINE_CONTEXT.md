# Pipeline Context

This file is a working reference for answering follow-up questions about the
current repository.

## Task

The project predicts the next web UI action for each row.

- Input: `task`, `history`, `cleaned_html`, `candidate_elements`
- Output: `op`, `target_id`, `value`
- Valid ops: `CLICK`, `TYPE`, `SELECT`
- Exact match requires all three output fields to match.
- Each row has 15 candidate elements.

## Main Files

- `src/preprocess.py`
  - Shared rule/prior utilities.
  - Builds weak empirical priors that do not use history-flow templates.
  - Extracts values from task text.
  - Provides rule-based fallback.
  - Applies final consistency guards.
- `src/retrieval.py`
  - Builds a lightweight example retriever from train rows.
  - Retrieves similar examples by site, history signature, and task token overlap.
- `src/train.py`
  - Fine-tunes a LoRA adapter with Unsloth + TRL SFT.
  - Can run grouped validation by `site_token`.
  - Saves adapter to `lora_model/`.
- `src/inference.py`
  - Builds empirical priors and retrieval index from all train data.
  - Routes each test row through LLM or fallback.
  - Writes `submission.csv`.
- `diagnostic.py`
  - Legacy diagnostic for old template behavior and rough prompt length.
- `check.py`
  - Checks how often the submission picked the first/default candidate.

## Current Pipeline

```text
data/train.csv
  -> build_empirical_priors()
  -> ExampleRetriever().build()

data/test.csv rows
  -> LoRA LLM if lora_model exists
  -> fallback_rule_based() only if no model or invalid LLM target
  -> enforce_consistency()
  -> submission.csv
```

## Empirical Priors

Function: `build_empirical_priors(train_df)`

Priors are weak tie-break context for the LLM. They do not directly produce
answers and do not use `(site_token, history_signature)` next-step templates.

- Global op distribution.
- Global `P(op | tag)` style tag/op counts.
- Site-level `P(op | tag, site_token)` style tag/op counts.
- Site-level common `(op, tag, label)` summaries.

Prompt text explicitly says task and current candidate evidence override
history patterns or empirical priors when they conflict.

## LLM Stage

Model:

- Base: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- Adapter path: `lora_model/`
- Training/inference max sequence length: `2048`
- Inference batch size: `4`
- Generation is deterministic: `do_sample=False`, `max_new_tokens=128`

Prompt contains:

- Task
- History
- Similar past examples from retrieval
- All candidate elements with attrs
- A strict JSON-only output instruction

The LLM output is parsed with a JSON regex. If the produced `target_id` is not
one of the current row's candidates, inference falls back to rules.

## Retrieval

Function: `ExampleRetriever.query(row, k=2)`

Lookup order:

1. Same `(site_token, history_signature)`
2. Same `site_token`
3. Same `history_signature`

Within each bucket, examples are ranked by Jaccard similarity over task tokens.
Retrieved examples include target label/op/value, but the prompt explicitly
warns that example `target_id` values are not valid for the current row.

## Rule-Based Fallback

Function: `fallback_rule_based(row, candidates)`

Operation choice:

- `TYPE` if task matches word-boundary regex: `type|enter|input`
- `SELECT` if task matches word-boundary regex: `select|choose`
- Otherwise `CLICK`

Target choice:

- Restricts to compatible tags for `TYPE`/`SELECT`.
- Picks the candidate whose text/label words overlap most with task text.
- For generic `CLICK`, it also checks quoted click labels.

Value extraction:

Function: `extract_value_from_task(task, op, attrs)`

Priority:

1. `CLICK` always returns empty string.
2. For `SELECT`, match available `options=` against task text first.
3. Quoted string fallback.
4. For `TYPE`, use label/placeholder/aria-label and take following text chunk.
5. Date regex fallback: `YYYY-MM-DD`.
6. Empty string.

## Consistency Guard

Function: `enforce_consistency(pred, candidates)`

It repairs common invalid outputs:

- Invalid op -> `CLICK`
- Invalid `target_id` -> fallback rules
- `TYPE` on non-input/non-textarea -> switch to best input or downgrade
- `SELECT` on non-select -> switch to best select or downgrade
- `CLICK` on select with extractable option -> upgrade to `SELECT`
- `SELECT` value -> force to one available option
- `CLICK` value -> clear to empty string

Repair counts are tracked in `CONSISTENCY_DEBUG` and printed by inference.

## Training Flow

`src/train.py` defaults:

- `VALIDATION_MODE = True`
- `FINAL_TRAIN_ON_FULL_DATA = False`
- Group split by `site_token` with `test_size=0.2`
- Train rows build empirical priors and retrieval index.
- All train rows become SFT examples; template-solved row filtering was removed.
- Adapter is saved to `lora_model/`.
- Held-out evaluation writes `outputs/eval_metrics.json` if training completes.

SFT config:

- Model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
- LoRA rank `r=16`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- Batch size `1`
- Gradient accumulation `16`
- `max_steps=1000`
- Learning rate `2e-4`
- Optimizer `adamw_8bit`

## Inference Flow

`src/inference.py`:

1. Reads `data/train.csv`, `data/test.csv`, `data/somenna_submission.csv`.
2. Builds empirical priors from all train rows.
3. Builds retriever from all train rows.
4. Loads `lora_model/` if it exists.
5. Routes test rows:
   - all rows queue for LLM if model exists
   - otherwise rule fallback
6. Validates LLM `target_id`.
7. Re-extracts value from task and selected candidate attrs.
8. Applies consistency guard.
9. Merges with sample submission ID order.
10. Writes `submission.csv`.

## Current Workspace State Notes

- `submission.csv` and `submission_pre_patch.csv` already exist.
- `lora_model/` exists and contains a LoRA adapter.
- `outputs/checkpoint-100/` exists.
- `outputs/eval_metrics.json` was not present when this file was created.
- Git status showed local modifications in:
  - `src/inference.py`
  - `src/preprocess.py`
  - `src/train.py`
  - untracked `src/retrieval.py`
  - untracked `my_code.zip`
- Local `python` and `py` commands were not available in the shell environment,
  so CSV statistics were not recomputed during this review.

## Useful Answering Heuristics

- If asked about accuracy risks, focus first on value extraction errors,
  LLM target hallucination, weak-prior overuse, and unseen site behavior.
- If asked how to improve score, likely high-impact areas are:
  - Better validation split and measured ablations.
  - More precise value extraction for TYPE.
  - Better prompt/prior calibration.
  - Better candidate ranking for fallback.
  - Training final adapter on full data after validation decisions are fixed.
- If asked about final submission behavior, answer from `src/inference.py`, not
  only from `train.py`, because inference builds priors/retrieval from all train data.
