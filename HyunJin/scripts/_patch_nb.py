import json

nb_path = r"C:\Users\shjsh\Desktop\ai contest\HyunJin\result\colab_pipeline.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

cell7_src = (
    "## 4. GPU별 설정 자동 패치\n\n"
    "| GPU | VRAM | batch | grad_accum | 유효배치 | max_steps | lr | 학습시간 | inf batch |\n"
    "|-----|------|-------|------------|---------|-----------|-----|---------|----------|\n"
    "| **A100** | 40GB | 32 | 1 | 32 | 2400 | 4e-4 | ~2.5h | 32 |\n"
    "| L4  | 24GB | 2 | 8 | 16 | 1200 | 2e-4 | ~4.3h | 8 |\n"
    "| T4  | 16GB | 1 | 16 | 16 | 800  | 2e-4 | ~4.0h | 4 |\n\n"
    "**변경사항**: CoT(Chain-of-Thought) 적용 — 답변 앞에 추론 1줄 추가, max_new_tokens 64→128\n\n"
    "> **A100 권장**: 3시간 세션 안에 학습+추론 완결 가능\n"
)

cell9_src = (
    "## 5. 학습 (LoRA SFT + CoT)\n\n"
    "- 모델: `Qwen3-8B` (4bit)\n"
    "- LoRA: r=16, 7개 projection\n"
    "- max_steps: 2400, lr=4e-4, cosine scheduler\n"
    "- 학습 제외: `site_2aa627db`\n"
    "- **CoT**: 답변 = 추론 1줄 + JSON (real_web target_acc 개선 목적)\n\n"
    "산출물: `/content/lora_model/`, `/content/outputs/eval_metrics.json`\n"
)

nb["cells"][7]["source"] = [cell7_src]
nb["cells"][9]["source"] = [cell9_src]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("완료")
print("Cell 7 첫줄:", repr("".join(nb["cells"][7]["source"])[:60]))
print("Cell 9 첫줄:", repr("".join(nb["cells"][9]["source"])[:60]))
