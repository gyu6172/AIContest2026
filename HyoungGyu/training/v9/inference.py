# -*- coding: utf-8 -*-
import csv
import argparse
import json
import os
import re
from typing import Any

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

from collections import Counter

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
    rerank_candidates_by_embedding,
    RETRIEVAL_K,
)
from retrieval import ExampleRetriever


# ─────────────────────────────────────────────
# 런타임 설정
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH        = 4096    # T4/L4 Colab friendly default
BATCH_SIZE            = 4       # T4-safe inference batch
CSV_CHUNK_SIZE        = 512
USE_RETRIEVAL         = True
USE_CONSISTENCY       = True
OOF_N_FOLDS           = 3
EXPERIMENT_E3         = True

# ─────────────────────────────────────────────
# 추론 통계
# ─────────────────────────────────────────────
STATS = Counter()


def reset_stats():
    STATS.clear()


def get_stats():
    return dict(STATS)


# ─────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────

def _parse_json_safe(text: str) -> dict | None:
    """중괄호가 중첩되지 않는 마지막 JSON 블록부터 역순으로 파싱 시도.

    모델이 thinking 도중 { } 를 사용해도 맨 끝 JSON만 파싱되도록 보장.
    """
    candidates_json = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
    for m in reversed(candidates_json):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def extract_json_answers(responses: list[str],
                         candidates_list: list[list[dict]]) -> list[tuple[str, str, str]]:
    """LLM 응답 배치를 파싱하고 choice 번호를 candidate_id로 변환한다."""
    results = []
    for resp, candidates in zip(responses, candidates_list):
        try:
            # Qwen3 thinking: </think> 이후만 파싱 (없으면 전체에서 역순 탐색)
            search_text = resp.split('</think>', 1)[-1] if '</think>' in resp else resp
            data = _parse_json_safe(search_text)
            if data is None:
                results.append(("CLICK", "", ""))
                continue
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
        max_new_tokens  = 512,    # Qwen3 thinking + CoT + JSON
        use_cache       = True,
        do_sample       = False,
        pad_token_id    = tokenizer.eos_token_id,
    )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    responses     = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answers       = extract_json_answers(responses, [item["candidates"] for item in batch])

    records = []
    for item, (op, target_id, value), raw_resp in zip(batch, answers, responses):
        candidates = item["candidates"]
        valid_ids  = {str(c.get("candidate_id", "")) for c in candidates}
        row        = item["row"]

        STATS["total"] += 1
        if "</think>" in raw_resp:
            STATS["thinking_used"] += 1

        if str(target_id) not in valid_ids:
            # choice 변환 실패 → rule-based fallback
            pred = fallback_rule_based(row, candidates)
            STATS["llm_fallback_invalid_target"] += 1
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
            STATS["llm_success"] += 1

        pred["_task"] = row["task"]
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, candidates)
        records.append(_clean_submission_record(item["id"], pred))
    return records


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def _save_report(base_dir: str, final_sub):
    import json
    from datetime import datetime

    stats     = get_stats()
    guard     = get_consistency_debug()
    total     = stats.get("total", 1)
    llm_ok    = stats.get("llm_success", 0)
    llm_fb    = stats.get("llm_fallback_invalid_target", 0)
    think     = stats.get("thinking_used", 0)
    rule_only = stats.get("no_model_rule_only", 0)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_rows": total,
        "source": {
            "llm_success":              {"n": llm_ok,    "pct": f"{llm_ok/total:.1%}"},
            "llm_fallback_bad_target":  {"n": llm_fb,    "pct": f"{llm_fb/total:.1%}"},
            "no_model_rule_only":       {"n": rule_only, "pct": f"{rule_only/total:.1%}"},
        },
        "thinking_used": {"n": think, "pct": f"{think/max(llm_ok+llm_fb,1):.1%}"},
        "consistency_guard": {
            k: {"n": v, "pct": f"{v/total:.1%}"} for k, v in sorted(guard.items())
        },
        "op_dist": dict(final_sub["op"].value_counts()),
    }

    # 콘솔 출력
    print("\n" + "="*55)
    print("  INFERENCE REPORT")
    print("="*55)
    print(f"  Total rows        : {total}")
    print(f"  LLM success       : {llm_ok:>5}  ({llm_ok/total:.1%})")
    print(f"  LLM fallback      : {llm_fb:>5}  ({llm_fb/total:.1%})  ← target 무효")
    print(f"  Rule-only (no LLM): {rule_only:>5}  ({rule_only/total:.1%})")
    print(f"  Thinking used     : {think:>5}  ({think/max(llm_ok+llm_fb,1):.1%})  ← <think> 토큰 확인")
    print("-"*55)
    print("  Consistency Guard repairs:")
    if guard:
        for k, v in sorted(guard.items(), key=lambda x: -x[1]):
            print(f"    {k:<35}: {v:>4}  ({v/total:.1%})")
    else:
        print("    (없음)")
    print("="*55 + "\n")

    # JSON 저장
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    report_path = os.path.join(artifacts_dir, "inference_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved: {report_path}")


def main(e1_no_survivor_retry: bool = False, e3_compact_prompt: bool = EXPERIMENT_E3):
    global EXPERIMENT_E3
    EXPERIMENT_E3 = bool(e3_compact_prompt)
    base_dir        = os.path.dirname(os.path.abspath(__file__))
    train_path      = os.path.join(base_dir, "data", "train.csv")
    test_path       = os.path.join(base_dir, "data", "test.csv")
    sample_sub_path = os.path.join(base_dir, "data", "somenna_submission.csv")

    print("1. Preparing retriever from train data...")
    train_df  = pd.read_csv(train_path)
    retriever = ExampleRetriever().build(train_df) if USE_RETRIEVAL else None

    print("2. Loading LLM (LoRA bundle)...")
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

    print("3. Running streaming inference (Dual-flow)...")
    final_results    = []
    llm_batch        = []

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

            if model is None:
                pred = fallback_rule_based(row, candidates)
                pred["_task"] = row["task"]
                if USE_CONSISTENCY:
                    pred = enforce_consistency(pred, candidates)
                final_results.append(_clean_submission_record(q_id, pred))
                STATS["total"] += 1
                STATS["no_model_rule_only"] += 1
                continue

            html_type = detect_html_type(row)

            # [Flow A] Workflow: RAG + Progress Status
            if html_type == "workflow":
                prompt, prompt_candidates = build_prompt(
                    row,
                    candidates,
                    retriever=retriever,
                    k=RETRIEVAL_K,
                    compact_candidates=EXPERIMENT_E3,
                    return_candidates=True,
                )
                llm_batch.append({
                    "id":         q_id,
                    "text":       f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
                    "candidates": prompt_candidates,
                    "row":        row,
                })

            # [Flow B] Real Web: Rerank + Intensive Thinking Mode
            else:
                # Universal Reranking is now inside build_prompt, 
                # but we can apply additional logic here if needed.
                prompt, prompt_candidates = build_prompt(
                    row,
                    candidates,
                    retriever=retriever,
                    k=RETRIEVAL_K,
                    compact_candidates=EXPERIMENT_E3,
                    return_candidates=True,
                )
                llm_batch.append({
                    "id":         q_id,
                    "text":       f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
                    "candidates": prompt_candidates,
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

    _save_report(base_dir, final_sub)



def ensemble_main(e3_compact_prompt: bool = False):
    """3-Fold OOF 앙상블 추론: 각 fold 모델의 예측을 다수결로 결합."""
    from collections import Counter
    import torch
    global EXPERIMENT_E3
    EXPERIMENT_E3 = bool(e3_compact_prompt)

    base_dir        = os.path.dirname(os.path.abspath(__file__))
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

        print(f"\n2-{fold}. Loading fold {fold} LoRA bundle...")
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

            html_type = detect_html_type(row)

            # Dual-flow logic preserved in ensemble
            prompt, prompt_candidates = build_prompt(
                row,
                candidates,
                retriever=retriever,
                k=RETRIEVAL_K,
                compact_candidates=EXPERIMENT_E3,
                return_candidates=True,
            )
            llm_batch.append({
                "id":         q_id,
                "text":       f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
                "candidates": prompt_candidates,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", action="store_true", help="Run OOF ensemble inference mode.")
    parser.add_argument(
        "--e1",
        action="store_true",
        help="E1: for no-survivor tournament rows, retry once with op-fixed constraint.",
    )
    parser.add_argument(
        "--e3",
        action="store_true",
        help="E3: use compact candidate formatting in prompts.",
    )
    args = parser.parse_args()

    if args.ensemble:
        ensemble_main(e3_compact_prompt=args.e3)
    else:
        main(e1_no_survivor_retry=args.e1, e3_compact_prompt=args.e3)
