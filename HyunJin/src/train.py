import os
import torch
import pandas as pd
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

from preprocess import predict_by_template, build_site_templates, format_candidates_with_attrs

def prepare_training_data(df, templates):
    print("Preparing training data (JSON Format + Attrs + Filtering)...")
    instructions = []
    
    for _, row in df.iterrows():
        try:
            cands = json.loads(row['candidate_elements'])
        except: continue
        
        # 템플릿 필터링 정상화 (규칙이 풀 수 있는 건 학습 제외)
        pred, conf = predict_by_template(row, templates, cands)
        is_correct = False
        if pred and conf >= 0.7:
            if pred['op'] == row['op'] and pred['target_id'] == str(row['target_id']):
                is_correct = True
                
        if not is_correct:
            # Attrs를 포함하여 프롬프트 작성
            cands_text = format_candidates_with_attrs(cands)
            
            prompt = f"""You are a web UI automation expert. Analyze the current situation and predict the next action.

[Task]
{row['task']}

[History]
{row['history']}

[Candidate Elements]
{cands_text}

Output ONLY valid JSON in the exact format:
{{"op": "CLICK|TYPE|SELECT", "target_id": "element_id", "value": "input_value_if_any"}}"""

            # CLICK 시 value 제거, JSON 포맷 지정
            val = str(row['value']) if pd.notna(row['value']) else ""
            if row['op'] == 'CLICK': val = ""
            
            answer = json.dumps({"op": row['op'], "target_id": str(row['target_id']), "value": val})
            
            instructions.append({
                "instruction": prompt,
                "input": "",
                "output": answer
            })
    
    return instructions

def train():
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32, # CLAUDE.md 권장값
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'train.csv')
    df = pd.read_csv(train_path)
    
    template_df, llm_df = train_test_split(df, test_size=0.7, random_state=42)
    
    templates = build_site_templates(template_df)
    train_data = prepare_training_data(llm_df, templates)
    dataset = Dataset.from_list(train_data)

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)
        return { "text" : texts }
    
    dataset = dataset.map(formatting_prompts_func, batched = True)

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
            warmup_ratio = 0.1, # CLAUDE.md 권장 비율
            max_steps = 1000,   # CLAUDE.md 권장 스텝
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

if __name__ == "__main__":
    train()