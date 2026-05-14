import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset
from src3.preprocess import generate_src3_dpo_pairs

# 전역 설정
BASE_MODEL_ID = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
OUTPUT_DIR = "src3_lora_model"
MAX_SEQ_LENGTH = 4096

def train_src3_dpo():
    print(f"--- Starting src3 DPO Training with {BASE_MODEL_ID} ---")
    
    # 1. 모델 및 토크나이저 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )
    
    # LoRA 패치
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 2. 데이터 로드 및 DPO 쌍 생성
    train_df = pd.read_csv("data/train.csv")
    # site_2aa627db 제외 (bias 제거)
    train_df = train_df[train_df['site_token'] != 'site_2aa627db'].reset_index(drop=True)
    
    dpo_data = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating DPO Pairs"):
        pair = generate_src3_dpo_pairs(row)
        if pair:
            dpo_data.append(pair)
    
    dataset = Dataset.from_list(dpo_data)
    print(f"Total DPO samples: {len(dataset)}")

    # 3. DPO Trainer 설정
    PatchDPOTrainer() # Unsloth 최적화 패치
    
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None, # PEFT 사용 시 None으로 설정하면 자동 처리
        args = TrainingArguments(
            per_device_train_batch_size = 4, # A100 80GB라면 8~16까지 가능할 수 있음
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs = 1,
            learning_rate = 5e-6,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 42,
            output_dir = OUTPUT_DIR,
            report_to = "none",
        ),
        beta = 0.1, # DPO KL penalty 제어
        train_dataset = dataset,
        tokenizer = tokenizer,
        max_length = MAX_SEQ_LENGTH,
        max_prompt_length = MAX_SEQ_LENGTH // 2,
    )

    # 4. 학습 시작
    print("Training...")
    dpo_trainer.train()
    
    # 5. 저장
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method = "lora")
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_src3_dpo()
