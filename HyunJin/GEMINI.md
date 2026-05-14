# GEMINI.md — Project Instructions

This project is a **Web UI Action Prediction** system designed to predict the next action (`op`, `target_id`, `value`) of an AI agent based on a natural language task, interaction history, and HTML candidate elements.

## 🚀 Project Overview

-   **Goal:** High-accuracy prediction of `CLICK`, `TYPE`, and `SELECT` operations on web elements.
-   **Architecture:**
    -   **Stage 1 (Retrieval):** Uses `retrieval.py` to find similar past examples (RAG) to provide few-shot context.
    -   **Stage 2 (LLM):** A fine-tuned `Qwen2.5-3B-Instruct` model (using LoRA via Unsloth) predicts the action in JSON format.
    -   **Stage 3 (Consistency Guard):** Post-processes LLM output in `preprocess.py` to ensure valid `target_id` and operation consistency (e.g., `CLICK` should not have a value).
-   **Tech Stack:** Python, Unsloth, TRL (SFT/DPO), Pandas, BeautifulSoup4, lxml.

## 🛠️ Building and Running

### Training
-   **Standard Training:** `python src/train.py` (Uses GroupShuffleSplit).
-   **OOF (Out-Of-Fold) Training:** `python src/train.py --oof` (Trains 3-fold models for ensemble).
-   **DPO Mode:** Controlled by `USE_DPO` flag in `src/train.py`.

### Inference
-   **Standard Inference:** `python src/inference.py` (Uses the adapter in `lora_model/`).
-   **Ensemble Inference:** `python src/inference.py --ensemble` (Combines predictions from multiple OOF folds).

### Diagnostics & Utilities
-   **Prompt Diagnostics:** `python scripts/diagnostic.py` (Checks prompt length and target validity).
-   **Submission Check:** `python scripts/check.py` (Validates the generated `submission.csv`).

## 📜 Development Conventions

### Source Code Structure (`src/`)
-   `preprocess.py`: Contains core logic for prompt building, value extraction from tasks, rule-based fallbacks, and the `enforce_consistency` guard.
-   `retrieval.py`: Implements `ExampleRetriever` for finding similar tasks/histories.
-   `train.py`: Handles LoRA fine-tuning using Unsloth and TRL.
-   `inference.py`: Orchestrates the inference pipeline, including model loading and CSV generation.

### Key Practices
-   **JSON-Only Output:** The LLM is strictly instructed to output JSON. Use `_parse_json_safe` for robust parsing of model responses (handling `<think>` tags).
-   **Consistency Guard:** Always run predictions through `enforce_consistency(pred, candidates)` to fix common model hallucinations (e.g., predicting an invalid `target_id`).
-   **Data Augmentation:** During training, candidate elements are shuffled (`shuffle_n`) to prevent the model from over-relying on element order.
-   **Value Extraction:** Use `extract_value_from_task` for `TYPE` and `SELECT` operations, prioritizing explicit values in the task description.

### Documentation Mapping
-   **Master Index:** See `CLAUDE.md` for the most up-to-date technical index.
-   **Technical Docs:** Detailed explanations are located in the `디지털경진대회 \` folder (referred to as `docs/` in some scripts).

## ⚠️ Important Notes
-   **VRAM Constraints:** Optimized for **A100 (40GB VRAM)**. Recommended `BATCH_SIZE`: **64** for fast inference, **32** for tournament mode.
-   **Site Token Bias:** Some sites (e.g., `site_2aa627db`) may be excluded during training due to extreme bias or absence in the test set.
