# -*- coding: utf-8 -*-
import csv
import json
import os
import re
from typing import Any

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

from preprocess import (
    enforce_consistency,
    extract_value_from_task,
    fallback_rule_based,
    detect_html_type,
    extract_workflow_context,
    format_numbered_candidates,
    choice_to_candidate_id,
    get_consistency_debug,
    build_prompt,
    RETRIEVAL_K,
    rerank_candidates_by_embedding,
)
from retrieval import ExampleRetriever


# ─────────────────────────────────────────────
# 런타임 설정
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH        = 2048
BATCH_SIZE            = 32      # A100: 32, L4: 8, T4: 4
CSV_CHUNK_SIZE        = 256
USE_RETRIEVAL         = True
USE_CONSISTENCY       = True
OOF_N_FOLDS           = 3       # OOF fold 수 (train.py와 일치)


# ─────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────

def extract_json_answers(responses: list[str],
                         candidates_list: list[list[dict]]) -> list[tuple[str, str, str]]:
    """LLM 응답 배치를 파싱하고 choice 번호를 candidate_id로 변환한다."""
    results = []
    for resp, candidates in zip(responses, candidates_list):
        try:
            match = re.search(r'\{.*?\}', resp, re.DOTALL)
            if not match:
                results.append(("CLICK", "", ""))
                continue
            data  = json.loads(match.group(0))
            op    = str(data.get("op", "CLICK")).upper()
            if op not in ("CLICK", "TYPE", "SELECT"):
                op = "CLICK"
            value     = "" if op == "CLICK" else str(data.get("value", ""))
            choice    = data.get("choice", "")
            target_id = choice_to_candidate_id(choice, candidates)
            results.append((op, target_id, value))
        except Exception:
            results.append(("CLICK", "", ""))
    return results


# ─────────────────────────────────────────────
# 제출 레코드 정리
# ─────────────────────────────────────────────

def _clean_submission_record(q_id, result: dict[str, Any]) -> dict[str, str]:
    op = str(result.get("op", "CLICK"))
    if op not in {"CLICK", "TYPE", "SELECT"}:
        op = "CLICK"
    value = str(result.get("value", "")).replace("\n", " ").replace("\r", " ").strip()
    if op == "CLICK":
        value = ""
    return {
        "id":        q_id,
        "op":        op,
        "target_id": str(result.get("target_id", "")),
        "value":     value,
    }


# ─────────────────────────────────────────────
# LLM 배치 추론
# ─────────────────────────────────────────────

def _run_llm_batch(batch: list[dict[str, Any]], model, tokenizer) -> list[dict[str, str]]:
    if not batch:
        return []

    texts  = [item["text"] for item in batch]
    inputs = tokenizer(
        texts,
        return_tensors  = "pt",
        padding         = True,
        truncation      = True,
        max_length      = MAX_SEQ_LENGTH,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens  = 128,    # CoT 1줄 + JSON
        use_cache       = True,
        do_sample       = False,
        pad_token_id    = tokenizer.eos_token_id,
    )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    responses     = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answers       = extract_json_answers(responses, [item["candidates"] for item in batch])

    records = []
    for item, (op, target_id, value) in zip(batch, answers):
        candidates = item["candidates"]
        valid_ids  = {str(c.get("candidate_id", "")) for c in candidates}
        row        = item["row"]

        if str(target_id) not in valid_ids:
            # choice 변환 실패 → rule-based fallback
            pred = fallback_rule_based(row, candidates)
        else:
            matched = next(
                (c for c in candidates if str(c.get("candidate_id", "")) == str(target_id)),
                None,
            )
            attrs = str(matched.get("attrs", "")) if matched else ""

            # LLM이 생성한 value 우선, 비어있으면 rule 보조
            if value and op != "CLICK":
                final_value = value
            elif op != "CLICK":
                final_value = extract_value_from_task(row["task"], op, attrs)
            else:
                final_value = ""

            pred = {"op": op, "target_id": target_id, "value": final_value}

        pred["_task"] = row["task"]
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, candidates)
        records.append(_clean_submission_record(item["id"], pred))
    return records


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    base_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path      = os.path.join(base_dir, "data", "train.csv")
    test_path       = os.path.join(base_dir, "data", "test.csv")
    sample_sub_path = os.path.join(base_dir, "data", "somenna_submission.csv")

    print("1. Preparing retriever from train data...")
    train_df  = pd.read_csv(train_path)
    retriever = ExampleRetriever().build(train_df) if USE_RETRIEVAL else None

    print("2. Loading LLM...")
    lora_dir = os.path.join(base_dir, "lora_model")
    model, tokenizer = None, None
    if os.path.exists(lora_dir):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = lora_dir,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit  = True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side   = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        print("  lora_model not found — using rule-based fallback only")

    print("3. Running streaming inference...")
    final_results = []
    llm_batch     = []

    def flush_llm_batch():
        nonlocal llm_batch
        if model is None or not llm_batch:
            return
        final_results.extend(_run_llm_batch(llm_batch, model, tokenizer))
        llm_batch = []

    for chunk in tqdm(pd.read_csv(test_path, chunksize=CSV_CHUNK_SIZE), desc="Chunks"):
        id_col = "id" if "id" in chunk.columns else chunk.columns[0]
        for _, row in chunk.iterrows():
            q_id = row[id_col]
            try:
                candidates = json.loads(row["candidate_elements"])
            except Exception:
                candidates = []

            # real_web이면 임베딩 기준 재정렬 (workflow는 원본 순서 유지)
            if detect_html_type(row) == "real_web":
                candidates = rerank_candidates_by_embedding(str(row.get("task", "")), candidates)

            if model is None:
                pred = fallback_rule_based(row, candidates)
                pred["_task"] = row["task"]
                if USE_CONSISTENCY:
                    pred = enforce_consistency(pred, candidates)
                final_results.append(_clean_submission_record(q_id, pred))
                continue

            prompt = build_prompt(
                row, candidates,
                retriever=retriever,
                k=RETRIEVAL_K,
            )
            llm_batch.append({
                "id":         q_id,
                "text":       f"### Instruction:\n{prompt}\n\n### Response:\n",
                "candidates": candidates,
                "row":        row,
            })
            if len(llm_batch) >= BATCH_SIZE:
                flush_llm_batch()

    flush_llm_batch()

    print("4. Saving submission...")
    sub_df = pd.DataFrame(final_results)
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        final_sub  = sample_sub[["id"]].merge(sub_df, on="id", how="left")
    else:
        final_sub = sub_df

    final_sub.fillna("", inplace=True)
    final_sub.loc[final_sub["op"] == "CLICK", "value"] = ""
    final_sub.loc[~final_sub["op"].isin(["CLICK", "TYPE", "SELECT"]), "op"] = "CLICK"

    empty_count = (final_sub["target_id"].astype(str).str.strip() == "").sum()
    if empty_count:
        print(f"Warning: {empty_count} rows with empty target_id")

    out_path = os.path.join(base_dir, "submission.csv")
    final_sub.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved: {out_path}  ({len(final_sub)} rows)")
    print(f"op dist: {dict(final_sub['op'].value_counts())}")
    print(f"consistency repairs: {get_consistency_debug()}")


def ensemble_main():
    """3-Fold OOF 앙상블 추론: 각 fold 모델의 예측을 다수결로 결합."""
    from collections import Counter
    import torch

    base_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path      = os.path.join(base_dir, "data", "train.csv")
    test_path       = os.path.join(base_dir, "data", "test.csv")
    sample_sub_path = os.path.join(base_dir, "data", "somenna_submission.csv")

    print("1. Preparing retriever from train data...")
    train_df  = pd.read_csv(train_path)
    retriever = ExampleRetriever().build(train_df) if USE_RETRIEVAL else None

    # test 행 미리 로드
    test_df = pd.read_csv(test_path)
    id_col  = "id" if "id" in test_df.columns else test_df.columns[0]
    all_ids = test_df[id_col].tolist()

    fold_predictions = []

    for fold in range(OOF_N_FOLDS):
        lora_dir = os.path.join(base_dir, f"lora_model_fold_{fold}")
        if not os.path.exists(lora_dir):
            print(f"  Fold {fold} model not found at {lora_dir}, skipping")
            continue

        print(f"\n2-{fold}. Loading fold {fold} model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = lora_dir,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit   = True,
        )
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side    = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  Running inference for fold {fold}...")
        fold_results = []
        llm_batch    = []

        def flush_batch():
            nonlocal llm_batch
            if not llm_batch:
                return
            fold_results.extend(_run_llm_batch(llm_batch, model, tokenizer))
            llm_batch = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Fold {fold}"):
            q_id = row[id_col]
            try:
                candidates = json.loads(row["candidate_elements"])
            except Exception:
                candidates = []

            if detect_html_type(row) == "real_web":
                candidates = rerank_candidates_by_embedding(str(row.get("task", "")), candidates)

            prompt = build_prompt(row, candidates, retriever=retriever, k=RETRIEVAL_K)
            llm_batch.append({
                "id":         q_id,
                "text":       f"### Instruction:\n{prompt}\n\n### Response:\n",
                "candidates": candidates,
                "row":        row,
            })
            if len(llm_batch) >= BATCH_SIZE:
                flush_batch()

        flush_batch()

        # dict로 변환
        fold_dict = {}
        for rec in fold_results:
            fold_dict[rec["id"]] = {
                "op": rec["op"], "target_id": rec["target_id"], "value": rec["value"]
            }
        fold_predictions.append(fold_dict)
        print(f"  Fold {fold}: {len(fold_dict)} predictions")

        # GPU 메모리 해제
        del model, tokenizer
        torch.cuda.empty_cache()

    # 다수결 앙상블
    print(f"\n3. Majority vote ensemble ({len(fold_predictions)} folds)...")
    final_results   = []
    agreement_count = 0

    for q_id in all_ids:
        votes = []
        for fp in fold_predictions:
            if q_id in fp:
                v = fp[q_id]
                votes.append((v["op"], v["target_id"], v["value"]))

        if not votes:
            final_results.append({"id": q_id, "op": "CLICK", "target_id": "", "value": ""})
            continue

        winner = Counter(votes).most_common(1)[0]
        if winner[1] == len(fold_predictions):
            agreement_count += 1
        op, target_id, value = winner[0]
        final_results.append({
            "id": q_id, "op": op, "target_id": target_id, "value": value
        })

    print(f"  만장일치 비율: {agreement_count}/{len(all_ids)} "
          f"({agreement_count/max(len(all_ids),1):.1%})")

    # 저장
    print("4. Saving submission...")
    sub_df = pd.DataFrame(final_results)
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        final_sub  = sample_sub[["id"]].merge(sub_df, on="id", how="left")
    else:
        final_sub = sub_df

    final_sub.fillna("", inplace=True)
    final_sub.loc[final_sub["op"] == "CLICK", "value"] = ""
    final_sub.loc[~final_sub["op"].isin(["CLICK", "TYPE", "SELECT"]), "op"] = "CLICK"

    empty_count = (final_sub["target_id"].astype(str).str.strip() == "").sum()
    if empty_count:
        print(f"Warning: {empty_count} rows with empty target_id")

    out_path = os.path.join(base_dir, "submission.csv")
    final_sub.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved: {out_path}  ({len(final_sub)} rows)")
    print(f"op dist: {dict(final_sub['op'].value_counts())}")


if __name__ == "__main__":
    import sys
    if "--ensemble" in sys.argv:
        ensemble_main()
    else:
        main()
