# v9 training pipeline

This version is based on `hyunjin/src` and focuses on raw_web target_id
disambiguation. The default model/profile is set for Colab T4/L4:
`unsloth/Qwen3-8B-bnb-4bit`, batch size 1 for training, and conservative DPO
sequence lengths.

## What changed

1. Prompt-to-candidate order is now consistent.
   - `build_prompt(..., return_candidates=True)` returns both the prompt and the
     exact candidate list used to number the prompt.
   - `train.py` and `inference.py` map model `choice` back to `candidate_id`
     using that same returned list.
   - This fixes the previous risk where `build_prompt()` reranked candidates
     internally, while `choice_to_candidate_id()` still used the original order.

2. raw_web candidates now include best-effort HTML context.
   - `preprocess.py` parses `cleaned_html` with BeautifulSoup.
   - Each candidate line is always printed, even if DOM matching fails.
   - If a candidate can be matched back to HTML by `id`, `name`, `placeholder`,
     `aria-label`, `text`, or useful attrs such as `type` and `role`, v9 appends
     compact parent/sibling context.

3. Same-looking candidates are explicitly marked.
   - Candidates with the same `tag + text + attrs` signature are marked as
     `dup=current/total`.
   - When multiple matching DOM nodes exist, v9 assigns DOM occurrences as a
     best-effort hint: `dom=occurrence-best-effort`.
   - If no unique DOM match is possible, the prompt says `dom=no-unique-match`
     rather than silently pretending the candidate is distinguishable.

4. The old partial A-Tree omission issue is avoided.
   - The earlier tree formatter could omit candidates that failed DOM matching.
   - v9 uses `format_contextual_candidates()` for prompt candidates, so all 15
     candidates remain visible to the model.

## Important limitation

If two candidates have identical `tag`, `text`, and `attrs`, and neither has a
reliable DOM match in `cleaned_html`, the data does not contain enough explicit
information to perfectly separate them. v9 exposes that ambiguity to the model
with duplicate markers and uses task/history/context when available, but it
cannot recover a missing `candidate_id -> backend_node_id` link.

## Required files

Place these files under `HyoungGyu/training/v9/data/`:

- `train.csv`
- `test.csv`
- `somenna_submission.csv`

The code writes outputs under `HyoungGyu/training/v9/`:

- `lora_model/`
- `outputs/`
- `artifacts/`
- `submission.csv`

## Run order

From this folder:

```powershell
cd HyoungGyu\training\v9
python train.py
python inference.py
```

Optional OOF ensemble:

```powershell
python train.py --oof
python inference.py --ensemble
```

Optional compact prompt mode:

```powershell
python train.py --e3
python inference.py --e3
```

On Colab T4, prefer `--e3` if you hit OOM. If memory is still tight, reduce
`MAX_SEQ_LENGTH` in `train.py` and `inference.py` from `4096` to `2048`.
