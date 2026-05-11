import os
import torch
import pandas as pd
import json
import re
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

from preprocess import (
    build_empirical_priors,
    format_empirical_priors,
    format_candidates_with_attrs,
    fallback_rule_based,
    extract_value_from_task,
    enforce_consistency,
)
from retrieval import ExampleRetriever, format_similar_examples


# === Runtime flags / constants (rollback-friendly) ===
RETRIEVAL_K = 2
MAX_SEQ_LENGTH = 2048
EVAL_SAMPLE_SIZE = 300            # set to None for full eval
USE_RETRIEVAL = True
USE_CONSISTENCY = True
RETRIEVAL_TASK_CHARS = 200
RETRIEVAL_HISTORY_CHARS = 200
VALIDATION_MODE = True             # set False for final submission training
FINAL_TRAIN_ON_FULL_DATA = False   # set True to train the saved adapter on all rows


def build_prompt(row, candidates, retriever=None, priors=None, k=RETRIEVAL_K, exclude_id=None):
    cands_text = format_candidates_with_attrs(candidates)
    examples = retriever.query(row, k=k, exclude_id=exclude_id) if retriever else []
    examples_text = format_similar_examples(
        examples,
        max_task_chars=RETRIEVAL_TASK_CHARS,
        max_history_chars=RETRIEVAL_HISTORY_CHARS,
    )
    priors_text = format_empirical_priors(row, priors)
    return f"""You are a web UI automation expert. Analyze the current situation and predict the next action.

[Task]
{row['task']}

[History]
{row['history']}

{examples_text}

{priors_text}

[Candidate Elements]
{cands_text}

Note: The examples above are reference patterns. The target_id values from those examples DO NOT exist in the current candidates. You MUST select a target_id ONLY from the [Candidate Elements] list above.
Task and current candidate evidence must override history patterns or empirical priors when they conflict.

Output ONLY valid JSON in the exact format:
{{"op": "CLICK|TYPE|SELECT", "target_id": "element_id", "value": "input_value_if_any"}}"""


def extract_json_answer(response):
    try:
        m = re.search(r'\{.*?\}', response, re.DOTALL)
        if not m:
            return {'op': 'CLICK', 'target_id': '', 'value': ''}
        data = json.loads(m.group(0))
        op = str(data.get('op', 'CLICK')).upper()
        if op not in ('CLICK', 'TYPE', 'SELECT'):
            op = 'CLICK'
        value = "" if op == 'CLICK' else str(data.get('value', ''))
        return {'op': op, 'target_id': str(data.get('target_id', '')), 'value': value}
    except Exception:
        return {'op': 'CLICK', 'target_id': '', 'value': ''}


def prepare_training_data(df, retriever=None, priors=None):
    print("Preparing training data (JSON Format + Attrs + Retrieval + Priors)...")
    instructions = []

    for _, row in df.iterrows():
        try:
            cands = json.loads(row['candidate_elements'])
        except Exception:
            continue

        prompt = build_prompt(
            row, cands,
            retriever=(retriever if USE_RETRIEVAL else None),
            priors=priors,
            k=RETRIEVAL_K,
            exclude_id=row.get('id'),
        )

        val = str(row['value']) if pd.notna(row['value']) else ""
        if row['op'] == 'CLICK':
            val = ""

        answer = json.dumps({"op": row['op'], "target_id": str(row['target_id']), "value": val})

        instructions.append({
            "instruction": prompt,
            "input": "",
            "output": answer
        })

    return instructions


def evaluate_full_pipeline(model, tokenizer, val_df, retriever, priors, train_sites, base_dir, max_seq_length=MAX_SEQ_LENGTH):
    print("\n🧪 Running held-out LLM-first evaluation...")
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_n = len(val_df)
    if EVAL_SAMPLE_SIZE is not None and total_n > EVAL_SAMPLE_SIZE:
        val_df = val_df.sample(n=EVAL_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"📊 Sampled eval: {len(val_df)}/{total_n} rows (random_state=42)")
    else:
        print(f"📊 Full eval: {total_n} rows")

    rows = []
    y_true_ops, y_pred_ops = [], []

    for _, row in val_df.iterrows():
        try:
            candidates = json.loads(row['candidate_elements'])
        except Exception:
            candidates = []

        prompt = build_prompt(
            row, candidates,
            retriever=(retriever if USE_RETRIEVAL else None),
            priors=priors,
            k=RETRIEVAL_K,
            exclude_id=row.get('id'),
        )
        text = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pred = extract_json_answer(response)

        valid_ids = {str(c.get('candidate_id', '')) for c in candidates}
        if str(pred['target_id']) not in valid_ids:
            pred = fallback_rule_based(row, candidates)
        else:
            matched_cand = next((c for c in candidates if str(c.get('candidate_id', '')) == str(pred['target_id'])), None)
            c_attrs = str(matched_cand.get('attrs', '')) if matched_cand else ''
            pred['value'] = extract_value_from_task(row['task'], pred['op'], c_attrs)
        pred['_task'] = row['task']
        if USE_CONSISTENCY:
            pred = enforce_consistency(pred, candidates)

        true_value = "" if pd.isna(row.get('value', '')) else str(row.get('value', ''))
        if row['op'] == 'CLICK':
            true_value = ""

        item = {
            'id': row.get('id', ''),
            'site_seen': str(row.get('site_token', '')) in train_sites,
            'true_op': str(row['op']),
            'pred_op': str(pred['op']),
            'true_target_id': str(row['target_id']),
            'pred_target_id': str(pred['target_id']),
            'true_value': true_value,
            'pred_value': str(pred['value']),
        }
        item['op_correct'] = item['true_op'] == item['pred_op']
        item['target_correct'] = item['true_target_id'] == item['pred_target_id']
        item['value_correct'] = item['true_value'] == item['pred_value']
        item['exact_correct'] = item['op_correct'] and item['target_correct'] and item['value_correct']
        rows.append(item)
        y_true_ops.append(item['true_op'])
        y_pred_ops.append(item['pred_op'])

    eval_df = pd.DataFrame(rows)

    def summarize(frame):
        if len(frame) == 0:
            return {'n': 0, 'op_acc': None, 'target_id_acc': None, 'value_acc': None, 'exact_match': None}
        return {
            'n': int(len(frame)),
            'op_acc': float(frame['op_correct'].mean()),
            'target_id_acc': float(frame['target_correct'].mean()),
            'value_acc': float(frame['value_correct'].mean()),
            'exact_match': float(frame['exact_correct'].mean()),
        }

    metrics = {
        'overall': summarize(eval_df),
        'site_seen': summarize(eval_df[eval_df['site_seen']]),
        'site_unseen': summarize(eval_df[~eval_df['site_seen']]),
    }

    labels = ['CLICK', 'TYPE', 'SELECT']
    cm = confusion_matrix(y_true_ops, y_pred_ops, labels=labels)
    print("\nPredicted-op confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    print(f"\nEval metrics: {metrics['overall']}")

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def train():
    assert torch.cuda.is_available(), (
        "CUDA GPU not available. This pipeline requires a GPU runtime "
        "(Colab T4/L4 or local CUDA). CPU training is not supported."
    )
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ GPU: {gpu_name} | Total VRAM: {total_mem_gb:.1f} GB")

    max_seq_length = MAX_SEQ_LENGTH
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'train.csv')
    df = pd.read_csv(train_path)

    if VALIDATION_MODE and not FINAL_TRAIN_ON_FULL_DATA:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(df, groups=df['site_token']))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f"Grouped split: train_rows={len(train_df)}, val_rows={len(val_df)}, train_sites={train_df['site_token'].nunique()}, val_sites={val_df['site_token'].nunique()}")
    else:
        train_df = df.reset_index(drop=True)
        val_df = None
        print(f"Final training mode: train_rows={len(train_df)}, train_sites={train_df['site_token'].nunique()} (no held-out validation)")

    priors = build_empirical_priors(train_df)
    retriever = ExampleRetriever().build(train_df) if USE_RETRIEVAL else None
    train_data = prepare_training_data(train_df, retriever=retriever, priors=priors)
    dataset = Dataset.from_list(train_data)

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 16,
            warmup_ratio = 0.1,
            max_steps = 1000,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            save_steps = 200,
            save_total_limit = 3,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = os.path.join(base_dir, "outputs"),
        ),
    )

    print("\nStarting robust training (1000 steps, JSON output)...")
    trainer.train()

    model.save_pretrained(os.path.join(base_dir, "lora_model"))
    tokenizer.save_pretrained(os.path.join(base_dir, "lora_model"))
    print("\nTraining Complete! Saved to 'lora_model'.")

    if VALIDATION_MODE and val_df is not None:
        train_sites = set(train_df['site_token'].astype(str))
        evaluate_full_pipeline(model, tokenizer, val_df, retriever, priors, train_sites, base_dir, max_seq_length=max_seq_length)
    else:
        print("Skipped held-out evaluation because final training mode used all train.csv rows.")


if __name__ == "__main__":
    train()
