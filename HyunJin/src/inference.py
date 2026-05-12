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
    build_group_prompt,
    build_grounding_step1_prompt,
    RETRIEVAL_K,
)
from retrieval import ExampleRetriever


# ─────────────────────────────────────────────
# 런타임 설정
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH        = 2048
BATCH_SIZE            = 32      # A100: 32, L4: 8, T4: 4
CSV_CHUNK_SIZE        = 256
USE_RETRIEVAL         = False
USE_CONSISTENCY       = True
OOF_N_FOLDS           = 3       # OOF fold 수 (train.py와 일치)
EXPERIMENT_E1         = False   # no-survivor 시 op 고정 재질문
EXPERIMENT_E3         = False   # compact candidate prompt

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

def _batch_generate(texts: list[str], model, tokenizer) -> list[str]:
    """텍스트 리스트를 BATCH_SIZE 단위로 나눠 LLM 생성."""
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        results.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    return results


def _parse_step1_response(response: str) -> dict:
    """Step 1 응답 파싱 → {"op": ..., "action_desc": ...}. 실패 시 빈 dict."""
    try:
        search_text = response.split('</think>', 1)[-1] if '</think>' in response else response
        data = _parse_json_safe(search_text)
        if data is None:
            return {}
        op = str(data.get("op", "CLICK")).upper()
        if op not in ("CLICK", "TYPE", "SELECT"):
            op = "CLICK"
        return {"op": op, "action_desc": str(data.get("action_desc", ""))}
    except Exception:
        return {}


def _parse_group_winner(response: str, group_candidates: list):
    """그룹 프롬프트 응답 파싱 → winner dict 또는 None."""
    try:
        search_text = response.split("</think>", 1)[-1] if "</think>" in response else response
        data = _parse_json_safe(search_text)
        if data is None:
            return None
        choice = int(data.get("choice", 0))
        if choice == 0 or choice > len(group_candidates):
            return None
        candidate = group_candidates[choice - 1]
        op        = str(data.get("op", "CLICK")).upper()
        if op not in ("CLICK", "TYPE", "SELECT"):
            op = "CLICK"
        value = "" if op == "CLICK" else str(data.get("value", ""))
        return {"candidate": candidate, "op": op, "value": value}
    except Exception:
        return None


def _run_tournament_batch(batch: list[dict], model, tokenizer) -> list[dict]:
    """토너먼트 추론: real_web 행을 5개씩 그룹핑 → 2라운드로 최적 후보 선택.

    2단계 Grounding:
      Pass 0 (Step 1): task+history만으로 다음 액션 자연어 기술 → action_desc 수집
      Round 1: action_desc 주입 후 5개씩 그룹 평가
      Round 2: survivors 재평가
    """
    if not batch:
        return []

    from bs4 import BeautifulSoup

    # row별 soup 파싱 (컨텍스트 재사용)
    soups = []
    for item in batch:
        html_str = str(item["row"].get("cleaned_html", ""))
        try:
            soups.append(BeautifulSoup(html_str, "lxml") if html_str and html_str != "nan" else None)
        except Exception:
            soups.append(None)

    # ── Pass 0 (Step 1): 자연어 액션 기술 수집 ────────
    step1_texts = [
        f"### Instruction:\n{build_grounding_step1_prompt(item['row'])}\n\n### Response:\n<think>\n"
        for item in batch
    ]
    step1_responses  = _batch_generate(step1_texts, model, tokenizer)
    step1_results    = [_parse_step1_response(r) for r in step1_responses]
    action_descs     = [r.get("action_desc", "") for r in step1_results]
    STATS["grounding_step1_done"] += len(batch)
    STATS["grounding_step1_parsed"] += sum(1 for r in step1_results if r.get("action_desc"))

    # ── Round 1: 5개씩 그룹핑 (action_desc 주입) ──────
    r1_texts, r1_meta = [], []
    for row_idx, (item, soup) in enumerate(zip(batch, soups)):
        groups = [item["candidates"][i:i+5] for i in range(0, len(item["candidates"]), 5)]
        for group in groups:
            prompt = build_group_prompt(
                item["row"],
                group,
                soup,
                action_desc=action_descs[row_idx],
                compact_candidates=EXPERIMENT_E3,
            )
            r1_texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n")
            r1_meta.append((row_idx, group))

    r1_responses = _batch_generate(r1_texts, model, tokenizer)

    row_survivors: dict[int, list] = {i: [] for i in range(len(batch))}
    for (row_idx, group), resp in zip(r1_meta, r1_responses):
        winner = _parse_group_winner(resp, group)
        if winner:
            row_survivors[row_idx].append(winner)

    # ── Round 2: survivors 2개 이상인 행만 재처리 ─────
    r2_texts, r2_meta = [], []
    for row_idx, survivors in row_survivors.items():
        if len(survivors) >= 2:
            group_cands = [s["candidate"] for s in survivors]
            prompt = build_group_prompt(
                batch[row_idx]["row"],
                group_cands,
                soups[row_idx],
                action_desc=action_descs[row_idx],
                compact_candidates=EXPERIMENT_E3,
            )
            r2_texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n")
            r2_meta.append(row_idx)

    if r2_texts:
        r2_responses = _batch_generate(r2_texts, model, tokenizer)
        for row_idx, resp in zip(r2_meta, r2_responses):
            group_cands = [s["candidate"] for s in row_survivors[row_idx]]
            winner = _parse_group_winner(resp, group_cands)
            row_survivors[row_idx] = [winner] if winner else [row_survivors[row_idx][0]]

    # ── 최종 레코드 조합 ───────────────────────────────
    records            = [None] * len(batch)
    no_survivor_items  = []  # (row_idx, item) — B1 fallback 대상

    for row_idx, item in enumerate(batch):
        survivors  = row_survivors[row_idx]
        candidates = item["candidates"]
        row        = item["row"]

        STATS["total"] += 1

        if not survivors:
            # B1: survivor 없음 → 전체 15개 단발 LLM으로 재시도 (나중에 채움)
            no_survivor_items.append((row_idx, item))
            STATS["tournament_no_survivor"] += 1
            continue

        s         = survivors[0]
        target_id = str(s["candidate"].get("candidate_id", ""))
        valid_ids = {str(c.get("candidate_id", "")) for c in candidates}

        if target_id not in valid_ids:
            pred = fallback_rule_based(row, candidates)
            STATS["tournament_invalid_target"] += 1
        else:
            op, value = s["op"], s["value"]
            if not value and op != "CLICK":
                matched = next((c for c in candidates if str(c.get("candidate_id", "")) == target_id), None)
                attrs = str(matched.get("attrs", "")) if matched else ""
                value = extract_value_from_task(row["task"], op, attrs)
            pred = {"op": op, "target_id": target_id, "value": value}
            STATS["tournament_success"] += 1

        pred["_task"] = row["task"]
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, candidates)
        records[row_idx] = _clean_submission_record(item["id"], pred)

    # ── E1: no-survivor → op 고정 재질문 1회 ──────────
    pending_no_survivor = no_survivor_items
    if pending_no_survivor and EXPERIMENT_E1:
        e1_texts = []
        e1_items = []
        for row_idx, item in pending_no_survivor:
            fixed_op = step1_results[row_idx].get("op", "CLICK")
            prompt = build_prompt(
                item["row"],
                item["candidates"],
                action_desc=action_descs[row_idx],
                compact_candidates=EXPERIMENT_E3,
            )
            prompt += (
                f"\n\n[Constraint]\n"
                f"You MUST use op=\"{fixed_op}\" unless impossible. "
                f"Prioritize choosing the correct target element."
            )
            e1_texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n")
            e1_items.append((row_idx, item, fixed_op))

        e1_responses = _batch_generate(e1_texts, model, tokenizer)
        e1_answers = extract_json_answers(
            e1_responses, [item["candidates"] for _, item, _ in e1_items]
        )

        still_pending = []
        for (row_idx, item, fixed_op), (op, target_id, value), _ in zip(e1_items, e1_answers, e1_responses):
            candidates = item["candidates"]
            row = item["row"]
            valid_ids = {str(c.get("candidate_id", "")) for c in candidates}
            if str(target_id) not in valid_ids:
                still_pending.append((row_idx, item))
                continue
            if op != fixed_op:
                op = fixed_op
            if not value and op != "CLICK":
                matched = next(
                    (c for c in candidates if str(c.get("candidate_id", "")) == target_id), None
                )
                attrs = str(matched.get("attrs", "")) if matched else ""
                value = extract_value_from_task(row["task"], op, attrs)
            pred = {"op": op, "target_id": target_id, "value": value, "_task": row["task"]}
            if USE_CONSISTENCY:
                pred = enforce_consistency(pred, candidates)
            records[row_idx] = _clean_submission_record(item["id"], pred)
            STATS["tournament_ns_e1"] += 1
        pending_no_survivor = still_pending

    # ── B1: no-survivor → 전체 후보 단발 LLM ──────────
    if pending_no_survivor:
        ns_texts = []
        for row_idx, item in pending_no_survivor:
            prompt = build_prompt(
                item["row"],
                item["candidates"],
                action_desc=action_descs[row_idx],
                compact_candidates=EXPERIMENT_E3,
            )
            ns_texts.append(f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n")

        ns_responses = _batch_generate(ns_texts, model, tokenizer)
        ns_answers   = extract_json_answers(
            ns_responses, [item["candidates"] for _, item in pending_no_survivor]
        )

        for (row_idx, item), (op, target_id, value), raw_resp in zip(
            pending_no_survivor, ns_answers, ns_responses
        ):
            candidates = item["candidates"]
            row        = item["row"]
            valid_ids  = {str(c.get("candidate_id", "")) for c in candidates}

            if str(target_id) not in valid_ids:
                pred = fallback_rule_based(row, candidates)
                STATS["tournament_ns_rule"] += 1
            else:
                if not value and op != "CLICK":
                    matched = next(
                        (c for c in candidates if str(c.get("candidate_id", "")) == target_id), None
                    )
                    attrs = str(matched.get("attrs", "")) if matched else ""
                    value = extract_value_from_task(row["task"], op, attrs)
                pred = {"op": op, "target_id": target_id, "value": value}
                STATS["tournament_ns_llm"] += 1

            pred["_task"] = row["task"]
            if USE_CONSISTENCY:
                pred = enforce_consistency(pred, candidates)
            records[row_idx] = _clean_submission_record(item["id"], pred)

    return records


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
    g1_done = stats.get("grounding_step1_done", 0)
    g1_ok   = stats.get("grounding_step1_parsed", 0)
    if g1_done:
        print(f"  Grounding Step 1     : {g1_done:>5}  (parsed={g1_ok}, {g1_ok/max(g1_done,1):.1%})")
    t_ok    = stats.get("tournament_success", 0)
    t_ns    = stats.get("tournament_no_survivor", 0)
    t_inv   = stats.get("tournament_invalid_target", 0)
    t_ns_l  = stats.get("tournament_ns_llm", 0)
    t_ns_r  = stats.get("tournament_ns_rule", 0)
    print(f"  Tournament success    : {t_ok:>5}  ({t_ok/total:.1%})")
    print(f"  Tournament no-survivor: {t_ns:>5}  ({t_ns/total:.1%})")
    print(f"    └ ns→LLM  : {t_ns_l:>5}  ({t_ns_l/max(t_ns,1):.1%} of ns)")
    print(f"    └ ns→rule : {t_ns_r:>5}  ({t_ns_r/max(t_ns,1):.1%} of ns)")
    print(f"  Tournament bad target : {t_inv:>5}  ({t_inv/total:.1%})")
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


def main(e1_no_survivor_retry: bool = False, e3_compact_prompt: bool = False):
    global EXPERIMENT_E1, EXPERIMENT_E3
    EXPERIMENT_E1 = bool(e1_no_survivor_retry)
    EXPERIMENT_E3 = bool(e3_compact_prompt)
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
    final_results    = []
    llm_batch        = []   # workflow 전용
    tournament_batch = []   # real_web 전용

    def flush_llm_batch():
        nonlocal llm_batch
        if model is None or not llm_batch:
            return
        final_results.extend(_run_llm_batch(llm_batch, model, tokenizer))
        llm_batch = []

    def flush_tournament_batch():
        nonlocal tournament_batch
        if model is None or not tournament_batch:
            return
        final_results.extend(_run_tournament_batch(tournament_batch, model, tokenizer))
        tournament_batch = []

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

            if html_type == "real_web":
                # 토너먼트: 5개씩 그룹 → 2라운드
                tournament_batch.append({"id": q_id, "candidates": candidates, "row": row})
                if len(tournament_batch) >= BATCH_SIZE:
                    flush_tournament_batch()
            else:
                # workflow: 기존 단발 배치
                prompt = build_prompt(
                    row,
                    candidates,
                    retriever=retriever,
                    k=RETRIEVAL_K,
                    compact_candidates=EXPERIMENT_E3,
                )
                llm_batch.append({
                    "id":         q_id,
                    "text":       f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
                    "candidates": candidates,
                    "row":        row,
                })
                if len(llm_batch) >= BATCH_SIZE:
                    flush_llm_batch()

    flush_llm_batch()
    flush_tournament_batch()

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

            prompt = build_prompt(
                row,
                candidates,
                retriever=retriever,
                k=RETRIEVAL_K,
                compact_candidates=EXPERIMENT_E3,
            )
            llm_batch.append({
                "id":         q_id,
                "text":       f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n",
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
