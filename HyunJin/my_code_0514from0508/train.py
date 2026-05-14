# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import json
import re
import random
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import confusion_matrix

from preprocess import (
    fallback_rule_based,
    extract_value_from_task,
    enforce_consistency,
    detect_html_type,
    extract_workflow_context,
    format_numbered_candidates,
    choice_to_candidate_id,
    build_prompt,
    RETRIEVAL_K,
    rerank_candidates_by_embedding,
    generate_cot_reasoning,
)
from retrieval import ExampleRetriever


# ─────────────────────────────────────────────
# 런타임 플래그 (변경해야 할 설정은 여기서만)
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH            = 4096
EVAL_SAMPLE_SIZE          = 1000  # A100 80GB: 검증 샘플 확대
USE_RETRIEVAL             = False # RAG 비활성화 (사용자 요청)
USE_CONSISTENCY           = True
VALIDATION_MODE           = True  # False 로 바꾸면 전체 데이터 학습
FINAL_TRAIN_ON_FULL_DATA  = False # True 로 바꾸면 val 없이 전체 학습

# 학습 제외할 site_token (CLICK 편향 + test에 없음 — 팀원 분석)
EXCLUDE_SITES = {"site_2aa627db"}

# Augmentation & OOF
SHUFFLE_AUGMENT_N    = 2          # 원본 1 + 셔플 N = (1+N)배 데이터
OOF_N_FOLDS          = 3          # OOF fold 수


# ─────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────

def extract_json_answer(response, candidates):
    """LLM 응답에서 JSON을 파싱하고 choice 번호를 candidate_id로 변환한다."""
    try:
        search_text = response.split('</think>', 1)[-1] if '</think>' in response else response
        m = re.search(r'\{[^{}]*\}', search_text, re.DOTALL)
        if not m:
            return {"op": "CLICK", "target_id": "", "value": ""}
        data = json.loads(m.group(0))
        op = str(data.get("op", "CLICK")).upper()
        if op not in ("CLICK", "TYPE", "SELECT"):
            op = "CLICK"
        value     = "" if op == "CLICK" else str(data.get("value", ""))
        choice    = data.get("choice", "")
        target_id = choice_to_candidate_id(choice, candidates)
        return {"op": op, "target_id": target_id, "value": value}
    except Exception:
        return {"op": "CLICK", "target_id": "", "value": ""}


# ─────────────────────────────────────────────
# 학습 데이터 생성
# ─────────────────────────────────────────────

def prepare_training_data(df, retriever=None, shuffle_n=SHUFFLE_AUGMENT_N):
    """SFT용 (instruction, output) 쌍을 생성한다.

    변경 사항:
    - 출력 형식: {"op": ..., "choice": <1~15>, "value": ...}
    - target_id가 candidates에 없는 행은 skip (데이터 오류)
    - shuffle_n > 0이면 candidate 순서를 셔플한 augmented 예시 추가
    """
    print(f"Preparing training data (shuffle_n={shuffle_n})...")
    instructions = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            candidates = json.loads(row["candidate_elements"])
        except Exception:
            skipped += 1
            continue

        # real_web이면 임베딩 기준 재정렬 (workflow는 원본 순서 유지)
        if detect_html_type(row) == "real_web":
            candidates = rerank_candidates_by_embedding(str(row.get("task", "")), candidates)

        target_id = str(row.get("target_id", ""))
        choice = None
        for i, c in enumerate(candidates, 1):
            if str(c.get("candidate_id", "")) == target_id:
                choice = i
                break

        if choice is None:
            skipped += 1
            continue

        val = str(row["value"]) if pd.notna(row.get("value")) else ""
        if row["op"] == "CLICK":
            val = ""

        # ── 원본 예시 ──
        prompt = build_prompt(
            row, candidates,
            retriever=retriever,
            k=RETRIEVAL_K,
            exclude_id=row.get("id"),
        )
        target_cand = next((c for c in candidates if str(c.get("candidate_id", "")) == target_id), {})
        reasoning = generate_cot_reasoning(str(row.get("task", "")), row["op"], choice, target_cand, val)
        answer = f"<think>\n{reasoning}\n</think>\n" + json.dumps({"op": row["op"], "choice": choice, "value": val})
        instructions.append({"instruction": prompt, "input": "", "output": answer})

        # ── 셔플 augmentation (real_web은 2배 — workflow EM 이미 1.0) ──
        effective_shuffle = shuffle_n * (2 if detect_html_type(row) == "real_web" else 1)
        for _ in range(effective_shuffle):
            shuffled = candidates.copy()
            random.shuffle(shuffled)
            new_choice = None
            for i, c in enumerate(shuffled, 1):
                if str(c.get("candidate_id", "")) == target_id:
                    new_choice = i
                    break
            if new_choice is None:
                continue
            aug_prompt = build_prompt(
                row, shuffled,
                retriever=retriever,
                k=RETRIEVAL_K,
                exclude_id=row.get("id"),
            )
            aug_target_cand = next((c for c in shuffled if str(c.get("candidate_id", "")) == target_id), {})
            aug_reasoning = generate_cot_reasoning(str(row.get("task", "")), row["op"], new_choice, aug_target_cand, val)
            aug_answer = f"<think>\n{aug_reasoning}\n</think>\n" + json.dumps({"op": row["op"], "choice": new_choice, "value": val})
            instructions.append({"instruction": aug_prompt, "input": "", "output": aug_answer})

    print(f"  총 {len(instructions)}개 생성 (원본+셔플{shuffle_n}회, skip: {skipped}개)")
    return instructions


# ─────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────

def evaluate_full_pipeline(model, tokenizer, val_df, retriever, train_sites, base_dir,
                           max_seq_length=MAX_SEQ_LENGTH):
    print("\n🧪 Running held-out evaluation...")
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side  = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_n = len(val_df)
    if EVAL_SAMPLE_SIZE is not None and total_n > EVAL_SAMPLE_SIZE:
        val_df = val_df.sample(n=EVAL_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled eval: {len(val_df)}/{total_n} rows")

    rows = []
    y_true_ops, y_pred_ops = [], []

    for _, row in val_df.iterrows():
        try:
            candidates = json.loads(row["candidate_elements"])
        except Exception:
            candidates = []

        prompt  = build_prompt(row, candidates, retriever=retriever, k=RETRIEVAL_K,
                                exclude_id=row.get("id"))
        text    = f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n"
        inputs  = tokenizer([text], return_tensors="pt", padding=True, truncation=True,
                            max_length=max_seq_length).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        response  = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        pred      = extract_json_answer(response, candidates)

        # fallback: choice 변환 실패 시
        valid_ids = {str(c.get("candidate_id", "")) for c in candidates}
        if str(pred["target_id"]) not in valid_ids:
            pred = fallback_rule_based(row, candidates)
        else:
            matched = next((c for c in candidates
                            if str(c.get("candidate_id", "")) == str(pred["target_id"])), None)
            attrs = str(matched.get("attrs", "")) if matched else ""
            # LLM value 우선, 빈 문자열이면 rule 보조
            if not pred["value"] and pred["op"] != "CLICK":
                pred["value"] = extract_value_from_task(row["task"], pred["op"], attrs)

        pred["_task"] = row["task"]
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, candidates)

        true_value = "" if pd.isna(row.get("value", "")) else str(row.get("value", ""))
        if row["op"] == "CLICK":
            true_value = ""

        item = {
            "id": row.get("id", ""),
            "site_seen": str(row.get("site_token", "")) in train_sites,
            "html_type": detect_html_type(row),
            "true_op":        str(row["op"]),
            "pred_op":        str(pred["op"]),
            "true_target_id": str(row["target_id"]),
            "pred_target_id": str(pred["target_id"]),
            "true_value":     true_value,
            "pred_value":     str(pred["value"]),
        }
        item["op_correct"]     = item["true_op"]        == item["pred_op"]
        item["target_correct"] = item["true_target_id"] == item["pred_target_id"]
        item["value_correct"]  = item["true_value"]     == item["pred_value"]
        item["exact_correct"]  = item["op_correct"] and item["target_correct"] and item["value_correct"]
        rows.append(item)
        y_true_ops.append(item["true_op"])
        y_pred_ops.append(item["pred_op"])

    eval_df = pd.DataFrame(rows)

    def summarize(frame):
        if len(frame) == 0:
            return {"n": 0, "op_acc": None, "target_id_acc": None,
                    "value_acc": None, "exact_match": None}
        return {
            "n":            int(len(frame)),
            "op_acc":       float(frame["op_correct"].mean()),
            "target_id_acc": float(frame["target_correct"].mean()),
            "value_acc":    float(frame["value_correct"].mean()),
            "exact_match":  float(frame["exact_correct"].mean()),
        }

    metrics = {
        "overall":      summarize(eval_df),
        "site_seen":    summarize(eval_df[eval_df["site_seen"]]),
        "site_unseen":  summarize(eval_df[~eval_df["site_seen"]]),
        "workflow":     summarize(eval_df[eval_df["html_type"] == "workflow"]),
        "real_web":     summarize(eval_df[eval_df["html_type"] == "real_web"]),
    }

    labels = ["CLICK", "TYPE", "SELECT"]
    cm = confusion_matrix(y_true_ops, y_pred_ops, labels=labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print(f"\nOverall: {metrics['overall']}")
    print(f"Workflow: {metrics['workflow']}")
    print(f"Real-web: {metrics['real_web']}")

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


# ─────────────────────────────────────────────
# 학습 메인
# ─────────────────────────────────────────────

def train():
    assert torch.cuda.is_available(), (
        "CUDA GPU not available. GPU runtime (Colab T4/L4 or local CUDA) required."
    )
    gpu_name     = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name} | VRAM: {total_mem_gb:.1f} GB")

    max_seq_length = MAX_SEQ_LENGTH
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = "unsloth/Qwen3.6-35B-A3B-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit  = True,
    )
    tokenizer.padding_side   = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r               = 16,
        target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        lora_alpha      = 32,
        lora_dropout    = 0,
        bias            = "none",
        use_gradient_checkpointing = "unsloth",
        random_state    = 3407,
    )

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.csv")
    df         = pd.read_csv(train_path)

    # site_2aa627db 제외 (CLICK 편향 96.5% + test에 없음)
    before = len(df)
    df = df[~df["site_token"].isin(EXCLUDE_SITES)].reset_index(drop=True)
    print(f"Excluded sites {EXCLUDE_SITES}: {before} -> {len(df)} rows")

    if VALIDATION_MODE and not FINAL_TRAIN_ON_FULL_DATA:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(df, groups=df["site_token"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        print(f"Split: train={len(train_df)} rows / val={len(val_df)} rows "
              f"(train_sites={train_df['site_token'].nunique()}, "
              f"val_sites={val_df['site_token'].nunique()})")
    else:
        train_df = df.reset_index(drop=True)
        val_df   = None
        print(f"Final training mode: {len(train_df)} rows, "
              f"{train_df['site_token'].nunique()} sites")

    retriever  = ExampleRetriever().build(train_df) if USE_RETRIEVAL else None
    train_data = prepare_training_data(train_df, retriever=retriever)
    dataset    = Dataset.from_list(train_data)

    def formatting_prompts_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            texts.append(f"### Instruction:\n{instruction}\n\n### Response:\n{output}")
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = dataset,
        args = SFTConfig(
            dataset_text_field         = "text",
            max_seq_length             = max_seq_length,
            dataset_num_proc           = 4,
            padding_free               = False,
            per_device_train_batch_size = 32,
            gradient_accumulation_steps = 1,
            warmup_ratio               = 0.1,
            num_train_epochs           = 1,
            max_steps                  = -1,
            learning_rate              = 2e-4, # 32B 모델에 적합한 보수적 LR
            fp16                       = not torch.cuda.is_bf16_supported(),
            bf16                       = torch.cuda.is_bf16_supported(),
            logging_steps              = 10,
            save_strategy              = "epoch",
            save_total_limit           = 2,
            optim                      = "adamw_8bit",
            weight_decay               = 0.01,
            lr_scheduler_type          = "cosine",
            seed                       = 3407,
            output_dir                 = os.path.join(base_dir, "outputs"),
        ),
    )

    print("\nStarting training (1 epoch)...")
    trainer.train()

    model.save_pretrained(os.path.join(base_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(base_dir, "lora_model"))
    print("Training complete. Saved to 'lora_model'.")

    if VALIDATION_MODE and val_df is not None:
        train_sites = set(train_df["site_token"].astype(str))
        evaluate_full_pipeline(model, tokenizer, val_df, retriever,
                               train_sites, base_dir, max_seq_length)
    else:
        print("Skipped held-out evaluation (final training mode).")


def train_oof():
    """3-Fold OOF 학습: fold별로 LoRA 학습 → 검증 → 어댑터 저장."""
    assert torch.cuda.is_available(), "CUDA GPU required."
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name} | VRAM: {total_mem_gb:.1f} GB")
    print(f"OOF Mode: {OOF_N_FOLDS} folds, shuffle_n={SHUFFLE_AUGMENT_N}")

    max_seq_length = MAX_SEQ_LENGTH
    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train.csv")
    df         = pd.read_csv(train_path)

    before = len(df)
    df = df[~df["site_token"].isin(EXCLUDE_SITES)].reset_index(drop=True)
    print(f"Excluded sites {EXCLUDE_SITES}: {before} -> {len(df)} rows")

    gkf = GroupKFold(n_splits=OOF_N_FOLDS)
    all_oof_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["site_token"])):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{OOF_N_FOLDS - 1}")
        print(f"{'='*60}")

        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold   = df.iloc[val_idx].reset_index(drop=True)
        print(f"  train={len(train_fold)} ({train_fold['site_token'].nunique()} sites) / "
              f"val={len(val_fold)} ({val_fold['site_token'].nunique()} sites)")

        # 각 fold마다 모델을 새로 로드 (이전 fold 가중치 오염 방지)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = "unsloth/Qwen3.6-35B-A3B-bnb-4bit",
            max_seq_length = max_seq_length,
            load_in_4bit   = True,
        )
        tokenizer.padding_side    = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r               = 16,
            target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            lora_alpha      = 32,
            lora_dropout    = 0,
            bias            = "none",
            use_gradient_checkpointing = "unsloth",
            random_state    = 3407 + fold,
        )

        retriever  = ExampleRetriever().build(train_fold) if USE_RETRIEVAL else None
        train_data = prepare_training_data(train_fold, retriever=retriever)
        dataset    = Dataset.from_list(train_data)

        def formatting_prompts_func(examples):
            texts = []
            for instruction, output in zip(examples["instruction"], examples["output"]):
                texts.append(f"### Instruction:\n{instruction}\n\n### Response:\n{output}")
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)

        fold_output_dir = os.path.join(base_dir, f"outputs_fold_{fold}")
        trainer = SFTTrainer(
            model            = model,
            processing_class = tokenizer,
            train_dataset    = dataset,
            args = SFTConfig(
                dataset_text_field         = "text",
                max_seq_length             = max_seq_length,
                dataset_num_proc           = 4,
                padding_free               = False,
                per_device_train_batch_size = 32,
                gradient_accumulation_steps = 1,
                warmup_ratio               = 0.1,
                num_train_epochs           = 1,
                max_steps                  = -1,
                learning_rate              = 2e-4,
                fp16                       = not torch.cuda.is_bf16_supported(),
                bf16                       = torch.cuda.is_bf16_supported(),
                logging_steps              = 10,
                save_strategy              = "epoch",
                save_total_limit           = 1,
                optim                      = "adamw_8bit",
                weight_decay               = 0.01,
                lr_scheduler_type          = "cosine",
                seed                       = 3407 + fold,
                output_dir                 = fold_output_dir,
            ),
        )

        print(f"\n  Starting fold {fold} training (1 epoch)...")
        trainer.train()

        fold_model_dir = os.path.join(base_dir, f"lora_model_fold_{fold}")
        model.save_pretrained(fold_model_dir)
        tokenizer.save_pretrained(fold_model_dir)
        print(f"  Fold {fold} saved to '{fold_model_dir}'")

        # 검증
        train_sites = set(train_fold["site_token"].astype(str))
        metrics = evaluate_full_pipeline(
            model, tokenizer, val_fold, retriever,
            train_sites, base_dir, max_seq_length,
        )
        all_oof_metrics.append({"fold": fold, **metrics.get("overall", {})})

        # fold별 eval_metrics 별도 저장
        out_dir = os.path.join(base_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"eval_metrics_fold_{fold}.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # GPU 메모리 해제
        del model, tokenizer, trainer, dataset
        torch.cuda.empty_cache()
        print(f"  Fold {fold} complete. GPU memory cleared.")

    # OOF 전체 결과 요약
    print(f"\n{'='*60}")
    print("  OOF SUMMARY")
    print(f"{'='*60}")
    for m in all_oof_metrics:
        em = m.get('exact_match', 'N/A')
        print(f"  Fold {m['fold']}: EM={em}")

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "oof_summary.json"), "w") as f:
        json.dump(all_oof_metrics, f, indent=2)
    print("OOF training complete.")


if __name__ == "__main__":
    import sys
    if "--oof" in sys.argv:
        train_oof()
    else:
        train()
