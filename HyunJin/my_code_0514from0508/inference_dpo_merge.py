# -*- coding: utf-8 -*-
"""
inference_dpo_merge.py
======================
기존 submission에서 DPO 라우팅 대상(real_web + attrs)만 교체합니다.
DPO 모델만 로드하므로 VRAM 28GB만 사용 → 빠르고 안전.

사용법 (Colab):
  !python /content/my_code_0514from0508/inference_dpo_merge.py \
      --base_submission /content/submission_7.csv \
      --output /content/submission_dpo_merged.csv
"""
import os
import sys
import json
import re
import argparse
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unsloth import FastLanguageModel
from preprocess import (
    build_prompt,
    detect_html_type,
    rerank_candidates_by_embedding,
    enforce_consistency,
    extract_value_from_task,
    choice_to_candidate_id,
    fallback_rule_based,
)

MAX_SEQ_LENGTH  = 4096
BATCH_SIZE      = 128   # DPO 1개만 로드(28GB) → 잔여 52GB로 배치 128 가능
MAX_NEW_TOKENS  = 256


def has_meaningful_attrs(c):
    text  = str(c.get("text",  "")).strip()
    attrs = str(c.get("attrs", "")).strip()
    return bool(text or (attrs and attrs.lower() not in ("nan", "none", "")))


def extract_json_answers(responses, candidates_list):
    results = []
    for resp, candidates in zip(responses, candidates_list):
        try:
            search = resp.split("</think>", 1)[-1] if "</think>" in resp else resp
            m = re.search(r"\{[^{}]*\}", search, re.DOTALL)
            if not m:
                results.append(("CLICK", "", ""))
                continue
            data = json.loads(m.group(0))
            op   = str(data.get("op", "CLICK")).upper()
            if op not in ("CLICK", "TYPE", "SELECT"):
                op = "CLICK"
            val    = "" if op == "CLICK" else str(data.get("value", ""))
            choice = data.get("choice", "")
            tid    = choice_to_candidate_id(choice, candidates)
            results.append((op, tid, val))
        except Exception:
            results.append(("CLICK", "", ""))
    return results


def run_batch(items, model, tokenizer):
    if not items:
        return []
    texts  = [item["text"] for item in items]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_SEQ_LENGTH
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids   = outputs[:, inputs["input_ids"].shape[1]:]
    responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    answers   = extract_json_answers(responses, [item["candidates"] for item in items])

    records = []
    for item, (op, tid, val) in zip(items, answers):
        candidates = item["candidates"]
        row        = item["row"]
        matched    = next((c for c in candidates if str(c.get("candidate_id","")) == str(tid)), None)
        attrs      = str(matched.get("attrs","")) if matched else ""

        if val and op != "CLICK":
            final_val = val
        elif op != "CLICK":
            final_val = extract_value_from_task(row["task"], op, attrs)
        else:
            final_val = ""

        pred = {"op": op, "target_id": tid, "value": final_val, "_task": row["task"]}
        pred = enforce_consistency(pred, candidates)

        records.append({
            "id":        item["id"],
            "op":        pred["op"],
            "target_id": pred["target_id"],
            "value":     pred["value"],
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_submission", required=True, help="Path to existing good submission CSV")
    parser.add_argument("--output", default=None, help="Output path for merged submission")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    base_dir     = os.path.dirname(os.path.abspath(__file__))
    data_dir     = os.path.join(os.path.dirname(base_dir), "data")
    test_path    = os.path.join(data_dir, "test.csv")
    lora_dpo_dir = os.path.join(base_dir, "lora_model_dpo")
    output_path  = args.output or os.path.join(base_dir, "submission_dpo_merged.csv")

    assert os.path.exists(lora_dpo_dir), f"lora_model_dpo/ not found: {lora_dpo_dir}"
    assert os.path.exists(args.base_submission), f"Base submission not found: {args.base_submission}"

    # ── 1. 기존 submission 로드 ─────────────────────────────────
    print(f"\n1. Loading base submission: {args.base_submission}")
    base_sub = pd.read_csv(args.base_submission)
    print(f"   Rows: {len(base_sub)}")

    # ── 2. DPO 모델만 로드 (SFT 안 올림 → VRAM 절약!) ──────────
    print("\n2. Loading DPO model ONLY ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_dpo_dir, max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16, load_in_4bit=False
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"   DPO 로드 후 잔여 VRAM: {free:.1f} GB")

    # ── 3. DPO 라우팅 대상만 식별 ───────────────────────────────
    print("\n3. Identifying DPO-routed rows ...")
    test_df = pd.read_csv(test_path)
    id_col  = "id" if "id" in test_df.columns else test_df.columns[0]

    dpo_items = []
    dpo_ids   = set()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Routing"):
        q_id = row[id_col]
        try:
            candidates = json.loads(row["candidate_elements"])
        except Exception:
            continue

        if detect_html_type(row) != "real_web":
            continue

        candidates = rerank_candidates_by_embedding(str(row.get("task", "")), candidates)

        if any(has_meaningful_attrs(c) for c in candidates):
            prompt = build_prompt(row, candidates, retriever=None, k=0)
            text   = f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n"
            dpo_items.append({"id": q_id, "row": row, "candidates": candidates, "text": text})
            dpo_ids.add(q_id)

    print(f"   DPO 대상: {len(dpo_items)}건 (나머지 {len(test_df) - len(dpo_items)}건은 기존 submission 유지)")

    # ── 4. DPO 추론 ─────────────────────────────────────────────
    print("\n4. DPO inference ...")
    dpo_results = {}
    for i in tqdm(range(0, len(dpo_items), BATCH_SIZE), desc="DPO Batches"):
        batch = dpo_items[i:i + BATCH_SIZE]
        recs  = run_batch(batch, model, tokenizer)
        for rec in recs:
            dpo_results[rec["id"]] = rec

    del model
    torch.cuda.empty_cache()

    # ── 5. 병합: 기존 submission + DPO 결과 ─────────────────────
    print("\n5. Merging ...")
    replaced = 0
    rows_out = []
    for _, row in base_sub.iterrows():
        q_id = row["id"]
        if q_id in dpo_results:
            rec = dpo_results[q_id]
            rows_out.append({"id": q_id, "op": rec["op"], "target_id": rec["target_id"], "value": rec["value"]})
            replaced += 1
        else:
            rows_out.append({"id": q_id, "op": row["op"], "target_id": row["target_id"],
                             "value": row["value"] if pd.notna(row["value"]) else ""})

    pd.DataFrame(rows_out).to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"  기존 유지: {len(rows_out) - replaced}건")
    print(f"  DPO 교체: {replaced}건")
    print(f"  저장: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
