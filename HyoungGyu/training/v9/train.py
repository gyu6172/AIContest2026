# -*- coding: utf-8 -*-
import os
import argparse
import inspect
import torch
import pandas as pd
import json
import re
import random
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
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
    generate_cot_reasoning,
    hard_negative_shuffle,
    find_lca_hard_negative,
)
from retrieval import ExampleRetriever


# ─────────────────────────────────────────────
# 런타임 플래그 (변경해야 할 설정은 여기서만)
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH            = 4096  # T4/L4 Colab friendly default
EVAL_SAMPLE_SIZE          = 1000  # 검증 샘플 확대
USE_RETRIEVAL             = True
USE_CONSISTENCY           = True
VALIDATION_MODE           = True
FINAL_TRAIN_ON_FULL_DATA  = False
USE_DPO                   = True

# 학습 제외할 site_token
EXCLUDE_SITES = {"site_2aa627db"}

# Augmentation & OOF
SHUFFLE_AUGMENT_N    = 1
OOF_N_FOLDS          = 3

# Valid Unsloth 4-bit model. 35B variants do not fit on Colab T4.
BASE_MODEL_ID = "unsloth/Qwen3-8B-bnb-4bit"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_R = 16
LORA_ALPHA = 32

DPO_PER_DEVICE_BATCH   = 1
DPO_GRAD_ACCUM_STEPS   = 16

SFT_PER_DEVICE_BATCH   = 1
SFT_GRAD_ACCUM_STEPS   = 16

DATALOADER_NUM_WORKERS = 2
SFT_DATASET_NUM_PROC   = 2


def configure_a100_throughput() -> None:
    """TF32로 matmul·conv 가속 (A100)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_base_model_and_tokenizer(max_seq_length: int):
    """bnb 4bit 단일 GPU 적재. device_map 고정으로 CPU 오프로드·bnb 검증 오류 방지."""
    return FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL_ID,
        max_seq_length = max_seq_length,
        load_in_4bit   = True,
        device_map     = {"": 0},
    )


def prepare_tokenizer(tokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def attach_lora(model, random_state: int):
    return FastLanguageModel.get_peft_model(
        model,
        r               = LORA_R,
        target_modules  = LORA_TARGET_MODULES,
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = 0,
        bias            = "none",
        use_gradient_checkpointing = "unsloth",
        random_state    = random_state,
    )


def _dpo_training_kwargs():
    return dict(
        per_device_train_batch_size = DPO_PER_DEVICE_BATCH,
        gradient_accumulation_steps = DPO_GRAD_ACCUM_STEPS,
        dataloader_num_workers      = DATALOADER_NUM_WORKERS,
        dataloader_pin_memory     = True,
        dataloader_persistent_workers = DATALOADER_NUM_WORKERS > 0,
    )


def _sft_training_kwargs():
    return dict(
        per_device_train_batch_size = SFT_PER_DEVICE_BATCH,
        gradient_accumulation_steps = SFT_GRAD_ACCUM_STEPS,
        dataloader_num_workers      = DATALOADER_NUM_WORKERS,
        dataloader_pin_memory     = True,
        dataloader_persistent_workers = DATALOADER_NUM_WORKERS > 0,
    )


def make_dpo_config(**kwargs):
    """Build DPOConfig across TRL/Unsloth versions with slightly different args."""
    supported = set(inspect.signature(DPOConfig.__init__).parameters)
    supported.discard("self")
    dropped = sorted(k for k in kwargs if k not in supported)
    if dropped:
        print(f"DPOConfig: ignored unsupported args for this TRL version: {dropped}")
    return DPOConfig(**{k: v for k, v in kwargs.items() if k in supported})


# ─────────────────────────────────────────────
# JSON 파싱
# ─────────────────────────────────────────────

def _parse_json_safe(text: str) -> dict | None:
    """비중첩 JSON 블록을 역순으로 파싱 시도 (thinking 내 중괄호 오파싱 방지)."""
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def extract_json_answer(response, candidates):
    """LLM 응답에서 JSON을 파싱하고 choice 번호를 candidate_id로 변환한다."""
    try:
        search_text = response.split('</think>', 1)[-1] if '</think>' in response else response
        data = _parse_json_safe(search_text)
        if data is None:
            return {"op": "CLICK", "target_id": "", "value": ""}
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

def prepare_training_data(
    df,
    retriever=None,
    shuffle_n=SHUFFLE_AUGMENT_N,
    realweb_aug_boost: int = 1,
    compact_prompt: bool = False,
):
    """DPO 또는 SFT용 학습 데이터를 생성한다.

    WEPO 프레임워크 적용 (DPO Mode):
    1. Chosen: 정답 요소 (aw)
    2. Rejected: LCA 거리가 가장 가까운 오답 요소 (al).
       - 변별력 극대화를 위해 op/value는 chosen과 동일하게 유지 (요소 선택에만 집중).
    3. f_op 휴리스틱: 정답이 TYPE/SELECT면 오답을 33% 확률로 CLICK으로 변조하여 기능적 변별력 추가.
    """
    mode_str = "DPO" if USE_DPO else "SFT"
    print(
        f"Preparing training data ({mode_str} Mode, shuffle_n={shuffle_n}, "
        f"realweb_aug_boost={realweb_aug_boost}, compact_prompt={compact_prompt})..."
    )
    instructions = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            candidates = json.loads(row["candidate_elements"])
        except Exception:
            skipped += 1
            continue

        target_id = str(row.get("target_id", ""))
        chosen_val = str(row["value"]) if pd.notna(row.get("value")) else ""
        if row["op"] == "CLICK":
            chosen_val = ""

        # HTML 파싱 (A-Tree 및 CoT 용)
        from bs4 import BeautifulSoup
        html_str = str(row.get("cleaned_html", ""))
        soup = None
        if html_str and html_str != "nan":
            try:
                soup = BeautifulSoup(html_str, "lxml")
            except Exception:
                soup = None

        # ── 공통 Prompt ──
        prompt, prompt_candidates = build_prompt(
            row, candidates,
            retriever=retriever,
            k=RETRIEVAL_K,
            exclude_id=row.get("id"),
            compact_candidates=compact_prompt,
            use_rerank=False,
            return_candidates=True,
        )

        # ── 1. Chosen Action ──
        chosen_choice = None
        for i, c in enumerate(prompt_candidates, 1):
            if str(c.get("candidate_id", "")) == target_id:
                chosen_choice = i
                break

        if chosen_choice is None:
            skipped += 1
            continue

        target_cand = prompt_candidates[chosen_choice - 1]
        chosen_reasoning = generate_cot_reasoning(
            str(row.get("task", "")), row["op"], chosen_choice, target_cand, chosen_val, 
            candidates=prompt_candidates, soup=soup
        )
        chosen_output = f"<think>\n{chosen_reasoning}\n</think>\n" + json.dumps(
            {"op": row["op"], "choice": chosen_choice, "value": chosen_val}
        )

        if USE_DPO:
            # ── 2. Rejected Action (LCA 기반) ──
            rejected_cand = find_lca_hard_negative(row, prompt_candidates, target_id)
            if not rejected_cand:
                skipped += 1
                continue

            rejected_id = str(rejected_cand.get("candidate_id", ""))
            rejected_choice = next(i for i, c in enumerate(prompt_candidates, 1) if str(c.get("candidate_id", "")) == rejected_id)

            # f_op 휴리스틱: TYPE/SELECT 시 33% 확률로 CLICK 변조
            rejected_op = row["op"]
            rejected_val = chosen_val
            if row["op"] in ("TYPE", "SELECT") and random.random() < 0.33:
                rejected_op = "CLICK"
                rejected_val = ""

            rejected_reasoning = generate_cot_reasoning(
                str(row.get("task", "")), rejected_op, rejected_choice, rejected_cand, rejected_val, 
                candidates=prompt_candidates, soup=soup
            )
            rejected_output = f"<think>\n{rejected_reasoning}\n</think>\n" + json.dumps(
                {"op": rejected_op, "choice": rejected_choice, "value": rejected_val}
            )

            instructions.append({
                "prompt":   f"### Instruction:\n{prompt}\n\n### Response:\n",
                "chosen":   chosen_output,
                "rejected": rejected_output
            })
        else:
            # SFT 모드
            instructions.append({
                "instruction": prompt,
                "input": "",
                "output": chosen_output
            })

        # ── 셔플 Augmentation (과적합 방지 핵심) ──
        # real_web은 구조 암기 방지를 위해 더 많은 셔플 적용
        realweb_mult = 3 if detect_html_type(row) == "real_web" else 1
        effective_shuffle = shuffle_n * realweb_mult
        
        for _ in range(effective_shuffle):
            shuffled = hard_negative_shuffle(prompt_candidates, target_id)
            new_chosen_choice = next(i for i, c in enumerate(shuffled, 1) if str(c.get("candidate_id", "")) == target_id)

            aug_prompt, shuffled_prompt_candidates = build_prompt(
                row, shuffled,
                retriever=retriever,
                k=RETRIEVAL_K,
                exclude_id=row.get("id"),
                compact_candidates=compact_prompt,
                use_rerank=False,
                return_candidates=True,
            )

            aug_chosen_cand = shuffled_prompt_candidates[new_chosen_choice - 1]
            aug_chosen_reasoning = generate_cot_reasoning(
                str(row.get("task", "")), row["op"], new_chosen_choice, aug_chosen_cand, chosen_val, 
                candidates=shuffled_prompt_candidates, soup=soup
            )
            aug_chosen_output = f"<think>\n{aug_chosen_reasoning}\n</think>\n" + json.dumps(
                {"op": row["op"], "choice": new_chosen_choice, "value": chosen_val}
            )

            if USE_DPO:
                new_rejected_choice = next(i for i, c in enumerate(shuffled_prompt_candidates, 1) if str(c.get("candidate_id", "")) == rejected_id)
                aug_rejected_cand = shuffled_prompt_candidates[new_rejected_choice - 1]
                aug_rejected_reasoning = generate_cot_reasoning(
                    str(row.get("task", "")), rejected_op, new_rejected_choice, aug_rejected_cand, rejected_val, 
                    candidates=shuffled_prompt_candidates, soup=soup
                )
                aug_rejected_output = f"<think>\n{aug_rejected_reasoning}\n</think>\n" + json.dumps(
                    {"op": rejected_op, "choice": new_rejected_choice, "value": rejected_val}
                )
                instructions.append({
                    "prompt":   f"### Instruction:\n{aug_prompt}\n\n### Response:\n",
                    "chosen":   aug_chosen_output,
                    "rejected": aug_rejected_output
                })
            else:
                instructions.append({
                    "instruction": aug_prompt,
                    "input": "",
                    "output": aug_chosen_output
                })

    print(f"  총 {len(instructions)}개 예시 생성 ({mode_str} 모드, skip: {skipped}개)")
    return instructions


# ─────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────

def evaluate_full_pipeline(
    model,
    tokenizer,
    val_df,
    retriever,
    train_sites,
    base_dir,
    max_seq_length=MAX_SEQ_LENGTH,
    compact_prompt: bool = False,
):
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

        prompt, prompt_candidates = build_prompt(
            row,
            candidates,
            retriever=retriever,
            k=RETRIEVAL_K,
            exclude_id=row.get("id"),
            compact_candidates=compact_prompt,
            return_candidates=True,
        )
        text    = f"### Instruction:\n{prompt}\n\n### Response:\n<think>\n"
        inputs  = tokenizer([text], return_tensors="pt", padding=True, truncation=True,
                            max_length=max_seq_length).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        response  = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        pred      = extract_json_answer(response, prompt_candidates)

        # fallback: choice 변환 실패 시
        valid_ids = {str(c.get("candidate_id", "")) for c in prompt_candidates}
        if str(pred["target_id"]) not in valid_ids:
            pred = fallback_rule_based(row, prompt_candidates)
        else:
            matched = next((c for c in prompt_candidates
                            if str(c.get("candidate_id", "")) == str(pred["target_id"])), None)
            attrs = str(matched.get("attrs", "")) if matched else ""
            # LLM value 우선, 빈 문자열이면 rule 보조
            if not pred["value"] and pred["op"] != "CLICK":
                pred["value"] = extract_value_from_task(row["task"], pred["op"], attrs)

        pred["_task"] = row["task"]
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, prompt_candidates)

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

def train(e2_realweb_boost: bool = False, e3_compact_prompt: bool = False):
    assert torch.cuda.is_available(), (
        "CUDA GPU not available. GPU runtime (Colab T4/L4 or local CUDA) required."
    )
    gpu_name     = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name} | VRAM: {total_mem_gb:.1f} GB")

    configure_a100_throughput()
    max_seq_length = MAX_SEQ_LENGTH
    model, tokenizer = load_base_model_and_tokenizer(max_seq_length)
    prepare_tokenizer(tokenizer)
    model = attach_lora(model, random_state=3407)

    base_dir   = os.path.dirname(os.path.abspath(__file__))
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
    realweb_boost = 2 if e2_realweb_boost else 1
    train_data = prepare_training_data(
        train_df,
        retriever=retriever,
        realweb_aug_boost=realweb_boost,
        compact_prompt=e3_compact_prompt,
    )
    dataset    = Dataset.from_list(train_data)

    # ── Trainer 설정 ──
    if USE_DPO:
        print(f"\nStarting DPO training (Qwen3-32B Unsloth 4bit, A100 80GB, 1 epoch, beta=0.1)...")
        trainer = DPOTrainer(
            model            = model,
            ref_model        = None, # PEFT 시 None 이면 자동으로 생성
            processing_class = tokenizer,
            train_dataset    = dataset,
            args = make_dpo_config(
                **_dpo_training_kwargs(),
                warmup_ratio               = 0.1,
                num_train_epochs           = 1,
                max_steps                  = -1,
                learning_rate              = 5e-5,
                fp16                       = False,
                bf16                       = True,
                logging_steps              = 10,
                save_strategy              = "epoch",
                save_total_limit           = 3,
                optim                      = "adamw_8bit",
                weight_decay               = 0.01,
                lr_scheduler_type          = "cosine",
                seed                       = 3407,
                output_dir                 = os.path.join(base_dir, "outputs"),
                beta                       = 0.1,
                max_prompt_length          = 1024,
                max_length                 = 2048,
            ),
        )
    else:
        print(f"\nStarting SFT training (Qwen3-32B Unsloth 4bit, A100 80GB, 1 epoch)...")
        # SFT 전용 포맷팅 (DPO는 필요 없음)
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
                **_sft_training_kwargs(),
                dataset_text_field         = "text",
                max_seq_length             = max_seq_length,
                dataset_num_proc           = SFT_DATASET_NUM_PROC,
                padding_free               = True,
                warmup_ratio               = 0.1,
                num_train_epochs           = 1,
                max_steps                  = -1,
                learning_rate              = 1e-4,
                fp16                       = False,
                bf16                       = True,
                logging_steps              = 20,
                save_strategy              = "epoch",
                save_total_limit           = 3,
                optim                      = "adamw_8bit",
                weight_decay               = 0.01,
                lr_scheduler_type          = "cosine",
                seed                       = 3407,
                output_dir                 = os.path.join(base_dir, "outputs"),
            ),
        )

    trainer.train()

    model.save_pretrained(os.path.join(base_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(base_dir, "lora_model"))
    print("Training complete. Saved to 'lora_model'.")

    if VALIDATION_MODE and val_df is not None:
        train_sites = set(train_df["site_token"].astype(str))
        evaluate_full_pipeline(
            model,
            tokenizer,
            val_df,
            retriever,
            train_sites,
            base_dir,
            max_seq_length,
            compact_prompt=e3_compact_prompt,
        )
    else:
        print("Skipped held-out evaluation (final training mode).")


def train_oof(e2_realweb_boost: bool = False, e3_compact_prompt: bool = False):
    """3-Fold OOF 학습: fold별로 LoRA 학습 → 검증 → 어댑터 저장."""
    assert torch.cuda.is_available(), "CUDA GPU required."
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {gpu_name} | VRAM: {total_mem_gb:.1f} GB")
    print(f"OOF Mode: {OOF_N_FOLDS} folds, shuffle_n={SHUFFLE_AUGMENT_N}")
    configure_a100_throughput()

    max_seq_length = MAX_SEQ_LENGTH
    base_dir   = os.path.dirname(os.path.abspath(__file__))
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
        model, tokenizer = load_base_model_and_tokenizer(max_seq_length)
        prepare_tokenizer(tokenizer)
        model = attach_lora(model, random_state=3407 + fold)

        retriever  = ExampleRetriever().build(train_fold) if USE_RETRIEVAL else None
        realweb_boost = 2 if e2_realweb_boost else 1
        train_data = prepare_training_data(
            train_fold,
            retriever=retriever,
            realweb_aug_boost=realweb_boost,
            compact_prompt=e3_compact_prompt,
        )
        dataset    = Dataset.from_list(train_data)

        fold_output_dir = os.path.join(base_dir, f"outputs_fold_{fold}")

        if USE_DPO:
            print(f"\n  Starting fold {fold} DPO (Qwen3-32B 4bit, A100 80GB, 1 epoch, beta=0.1)...")
            trainer = DPOTrainer(
                model            = model,
                ref_model        = None,
                processing_class = tokenizer,
                train_dataset    = dataset,
                args = make_dpo_config(
                    **_dpo_training_kwargs(),
                    warmup_ratio               = 0.1,
                    num_train_epochs           = 1,
                    max_steps                  = -1,
                    learning_rate              = 5e-5,
                    fp16                       = False,
                    bf16                       = True,
                    logging_steps              = 10,
                    save_strategy              = "epoch",
                    save_total_limit           = 2,
                    optim                      = "adamw_8bit",
                    weight_decay               = 0.01,
                    lr_scheduler_type          = "cosine",
                    seed                       = 3407 + fold,
                    output_dir                 = fold_output_dir,
                    beta                       = 0.1,
                    max_prompt_length          = 1024,
                    max_length                 = 2048,
                ),
            )
        else:
            print(f"\n  Starting fold {fold} SFT (Qwen3-32B 4bit, A100 80GB, 1 epoch)...")
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
                    **_sft_training_kwargs(),
                    dataset_text_field         = "text",
                    max_seq_length             = max_seq_length,
                    dataset_num_proc           = SFT_DATASET_NUM_PROC,
                    padding_free               = True,
                    warmup_ratio               = 0.1,
                    num_train_epochs           = 1,
                    max_steps                  = -1,
                    learning_rate              = 1e-4,
                    fp16                       = False,
                    bf16                       = True,
                    logging_steps              = 20,
                    save_strategy              = "epoch",
                    save_total_limit           = 2,
                    optim                      = "adamw_8bit",
                    weight_decay               = 0.01,
                    lr_scheduler_type          = "cosine",
                    seed                       = 3407 + fold,
                    output_dir                 = fold_output_dir,
                ),
            )

        trainer.train()

        fold_model_dir = os.path.join(base_dir, f"lora_model_fold_{fold}")
        model.save_pretrained(fold_model_dir)
        tokenizer.save_pretrained(fold_model_dir)
        print(f"  Fold {fold} saved to '{fold_model_dir}'")

        # 검증
        train_sites = set(train_fold["site_token"].astype(str))
        metrics = evaluate_full_pipeline(
            model,
            tokenizer,
            val_fold,
            retriever,
            train_sites,
            base_dir,
            max_seq_length,
            compact_prompt=e3_compact_prompt,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", action="store_true", help="Run OOF training mode.")
    parser.add_argument(
        "--e2",
        action="store_true",
        help="E2: boost real_web augmentation in training data generation.",
    )
    parser.add_argument(
        "--e3",
        action="store_true",
        help="E3: use compact candidate formatting in prompts.",
    )
    args = parser.parse_args()

    if args.oof:
        train_oof(e2_realweb_boost=args.e2, e3_compact_prompt=args.e3)
    else:
        train(e2_realweb_boost=args.e2, e3_compact_prompt=args.e3)
