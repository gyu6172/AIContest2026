# Web UI Action Prediction Project (HyunJin)

본 프로젝트는 웹 브라우저 환경에서 사용자의 지시에 따라 최적의 행동(Click, Type, Select)을 예측하는 AI 에이전트를 개발하는 것을 목표로 합니다.

## 📁 주요 문서 구조 (Documentation Structure)

모든 주요 문서는 `docs/` 폴더 내에 체계적으로 구조화되어 있습니다.

### 1. [Pipeline (파이프라인 설계)](./docs/pipeline/EXPLAIN.md)
*   **[EXPLAIN.md](./docs/pipeline/EXPLAIN.md)**: 전체 아키텍처, 토너먼트 추론 방식, 가드레일 로직 등 기술 상세 설명.
*   **[GPU_PROFILES.md](./docs/pipeline/GPU_PROFILES.md)**: 하드웨어 설정 및 VRAM 최적화 가이드.

### 2. [Research & Strategy (연구 및 전략)](./docs/research/STRATEGY.md)
*   **[STRATEGY.md](./docs/research/STRATEGY.md)**: 데이터 분석 결과, 실패 사례 분석 및 향후 개선 방향.
*   **[PAPER_ANALYZE.md](./docs/research/PAPER_ANALYZE.md)**: WEPO, Mind2Web, SeeAct 등 핵심 연구 논문 분석.

### 3. [Archive (기록)](./docs/archive/DAILY_LOG.md)
*   **[DAILY_LOG.md](./docs/archive/DAILY_LOG.md)**: 프로젝트 진행 과정 및 일일 업데이트 기록.

---

## 🚀 빠른 시작 (Quick Start)

개발 환경 설정 및 실행 방법은 **[CLAUDE.md](./CLAUDE.md)** 파일을 참조하십시오.

### 주요 명령어
- **학습**: `python src/train.py`
- **추론**: `python src/inference.py`
- **앙상블 추론**: `python src/inference.py --ensemble`

---

## 🛠️ 핵심 기술 스택
- **Model**: Qwen3-8B (LoRA SFT / DPO)
- **Framework**: Unsloth, TRL, FastLanguageModel
- **Parsing**: BeautifulSoup4, lxml
- **Strategy**: 2-Stage Grounding, Tournament Inference, Consistency Guard
