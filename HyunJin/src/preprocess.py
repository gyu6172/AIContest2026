# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
from collections import Counter
from typing import Any

CONSISTENCY_DEBUG = Counter()


# ─────────────────────────────────────────────
# 기본 파싱 유틸리티
# ─────────────────────────────────────────────

def parse_attrs_str(attrs_str: str) -> dict:
    result = {}
    if attrs_str is None or pd.isna(attrs_str):
        return result
    attrs_text = str(attrs_str).strip()
    if not attrs_text:
        return result
    pattern = re.compile(r'(?:^|\s*\|\s*)([A-Za-z0-9_:-]+)\s*=\s*(.*?)(?=\s*\|\s*[A-Za-z0-9_:-]+\s*=|$)')
    for match in pattern.finditer(attrs_text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key:
            result[key] = value
    return result


def get_history_signature(history_str):
    """History의 op 시퀀스를 시그니처로 사용하여 워크플로우 분기 구별."""
    if pd.isna(history_str) or not str(history_str).strip():
        return "START"
    ops = re.findall(r'->\s*(CLICK|TYPE|SELECT)', str(history_str))
    if not ops:
        return "STEP_" + str(len(re.findall(r'Step \d+:', str(history_str))) + 1)
    return ",".join(ops)


# ─────────────────────────────────────────────
# Value 추출
# ─────────────────────────────────────────────

def extract_value_from_task(task: str, op: str, attrs: str) -> str:
    """task 문장에서 TYPE/SELECT에 필요한 value를 추출한다.

    우선순위:
    1. SELECT: options= 중 task에 등장하는 옵션 (exact → fuzzy)
    2. 따옴표 패턴
    3. TYPE: label/placeholder 기반 뒤따르는 청크
    4. 날짜 regex
    5. 빈 문자열
    """
    op = str(op or 'CLICK').upper()
    if op == 'CLICK':
        return ""

    task_str  = "" if task is None or pd.isna(task) else str(task)
    attrs_str = "" if attrs is None or pd.isna(attrs) else str(attrs)

    try:
        if op == 'SELECT':
            options = _parse_options(attrs_str)
            task_lower = task_str.lower()
            for opt in options:
                if opt.lower() in task_lower:
                    return opt
            best_opt, best_score = None, 0.0
            task_words = _token_set(task_lower)
            for opt in options:
                opt_words = _token_set(opt)
                score = len(opt_words & task_words) / max(len(opt_words), 1)
                if score > best_score:
                    best_score, best_opt = score, opt
            if best_opt and best_score >= 0.5:
                return best_opt

        quoted = re.search(r'["\'"]([^"\']+)["\'"]', task_str)
        if quoted:
            return quoted.group(1).strip()

        if op == 'TYPE':
            attrs_dict = parse_attrs_str(attrs_str)
            field_labels = [
                attrs_dict.get('label', ''),
                attrs_dict.get('placeholder', ''),
                attrs_dict.get('aria-label', ''),
                attrs_dict.get('name', ''),
            ]
            for field_label in [x for x in field_labels if x]:
                label_pattern = re.escape(str(field_label).strip())
                match = re.search(label_pattern + r'\s*(?:is|as|to|:|-)??\s*([^,.;\n]+)', task_str, flags=re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"\'')
                    if value:
                        return value

        date_match = re.search(r'\d{4}-\d{2}-\d{2}', task_str)
        if date_match:
            return date_match.group(0)
    except Exception:
        return ""

    return ""


# ─────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────

def _candidate_label(c):
    label = str(c.get('text', '')).strip().lower()
    if not label:
        attrs_dict = parse_attrs_str(str(c.get('attrs', '')))
        label = str(
            attrs_dict.get('aria-label', '') or
            attrs_dict.get('placeholder', '') or
            attrs_dict.get('label', '')
        ).strip().lower()
    return label


def _parse_options(attrs: str):
    if attrs is None or pd.isna(attrs):
        return []
    opts_str = parse_attrs_str(str(attrs)).get('options', '')
    if not opts_str:
        return []
    return [o.strip() for o in opts_str.split(' / ') if o.strip()]


def _token_set(text: str):
    return set(re.findall(r'\w+', str(text).lower()))


def _candidate_match_score(candidate: dict[str, Any], task: str) -> float:
    task_lower = str(task).lower()
    attrs_dict = parse_attrs_str(str(candidate.get('attrs', '')))
    fields = [
        candidate.get('text', ''),
        attrs_dict.get('label', ''),
        attrs_dict.get('placeholder', ''),
        attrs_dict.get('aria-label', ''),
        attrs_dict.get('name', ''),
    ]
    score = 0.0
    for word in re.findall(r'\w+', ' '.join(str(f) for f in fields).lower()):
        if len(word) > 2 and word in task_lower:
            score += 1.0
    label_words = _token_set(' '.join(str(f) for f in fields))
    task_words  = _token_set(task)
    if label_words:
        score += len(label_words & task_words) / max(len(label_words), 1)
    return score


def _best_candidate(pool, task: str):
    if not pool:
        return None
    return max(pool, key=lambda c: (_candidate_match_score(c, task), str(c.get('candidate_id', ''))))


def _best_option_for_task(options, task: str, current_value: str = ""):
    if not options:
        return ""

    value_norm = str(current_value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == value_norm:
            return opt

    value_words = _token_set(current_value)
    best_opt, best_score = None, 0.0
    if value_words:
        for opt in options:
            opt_words = _token_set(opt)
            score = len(opt_words & value_words) / max(len(opt_words), 1)
            if score > best_score:
                best_score, best_opt = score, opt
        if best_opt and best_score >= 0.5:
            return best_opt

    task_words = _token_set(task)
    best_opt, best_score = options[0], -1.0
    for opt in options:
        opt_words = _token_set(opt)
        score = len(opt_words & task_words) / max(len(opt_words), 1)
        if score > best_score:
            best_score, best_opt = score, opt
    return best_opt or options[0]


# ─────────────────────────────────────────────
# Consistency Guard
# ─────────────────────────────────────────────

def reset_consistency_debug():
    CONSISTENCY_DEBUG.clear()


def get_consistency_debug():
    return dict(CONSISTENCY_DEBUG)


def enforce_consistency(pred, candidates):
    """최종 가드레일: op/tag/value 일관성을 강제한다.

    선택적 키 `_task`는 수리 점수 계산에 사용되며, 반환 전에 제거된다.
    """
    task = str(pred.get('_task', '')) if isinstance(pred, dict) else ''
    pred = dict(pred or {})
    pred.pop('_task', None)
    pred.pop('_row', None)

    valid_ops = {'CLICK', 'TYPE', 'SELECT'}
    op = str(pred.get('op', 'CLICK')).upper()
    if op not in valid_ops:
        op = 'CLICK'
        CONSISTENCY_DEBUG['bad_op_to_click'] += 1

    cand_map  = {str(c.get('candidate_id', '')): c for c in candidates}
    target_id = str(pred.get('target_id', ''))
    value     = "" if pd.isna(pred.get('value', '')) else str(pred.get('value', ''))

    if target_id not in cand_map:
        fb = fallback_rule_based({'task': task}, candidates)
        op, target_id, value = fb['op'], str(fb['target_id']), str(fb.get('value', ''))
        CONSISTENCY_DEBUG['invalid_target_repaired'] += 1

    chosen = cand_map.get(str(target_id))
    tag    = str(chosen.get('tag', '')).lower() if chosen else ''

    if op == 'TYPE' and tag not in {'input', 'textarea'}:
        replacement = _best_candidate(
            [c for c in candidates if str(c.get('tag', '')).lower() in {'input', 'textarea'}], task
        )
        if replacement:
            target_id = str(replacement.get('candidate_id', ''))
            chosen    = replacement
            tag       = str(chosen.get('tag', '')).lower()
            value     = extract_value_from_task(task, 'TYPE', str(chosen.get('attrs', ''))) or value
            CONSISTENCY_DEBUG['type_target_switched'] += 1
        else:
            op, value = 'CLICK', ''
            CONSISTENCY_DEBUG['type_downgraded_click'] += 1

    if op == 'SELECT' and tag != 'select':
        replacement = _best_candidate(
            [c for c in candidates if str(c.get('tag', '')).lower() == 'select'], task
        )
        if replacement:
            target_id = str(replacement.get('candidate_id', ''))
            chosen    = replacement
            tag       = str(chosen.get('tag', '')).lower()
            value     = extract_value_from_task(task, 'SELECT', str(chosen.get('attrs', ''))) or value
            CONSISTENCY_DEBUG['select_target_switched'] += 1
        else:
            op, value = 'CLICK', ''
            CONSISTENCY_DEBUG['select_downgraded_click'] += 1

    if op == 'CLICK' and tag == 'select':
        extracted = extract_value_from_task(task, 'SELECT', str(chosen.get('attrs', ''))) if chosen else ''
        if extracted:
            op, value = 'SELECT', extracted
            CONSISTENCY_DEBUG['click_select_upgraded'] += 1
        else:
            replacement = _best_candidate(
                [c for c in candidates if str(c.get('tag', '')).lower() in {'button', 'a'}], task
            )
            if replacement:
                target_id = str(replacement.get('candidate_id', ''))
                chosen    = replacement
                tag       = str(chosen.get('tag', '')).lower()
                CONSISTENCY_DEBUG['click_select_target_switched'] += 1

    if op == 'SELECT':
        options     = _parse_options(str(chosen.get('attrs', ''))) if chosen else []
        fixed_value = _best_option_for_task(options, task, value)
        if fixed_value != value:
            CONSISTENCY_DEBUG['select_value_repaired'] += 1
        value = fixed_value

    if op == 'CLICK':
        if value:
            CONSISTENCY_DEBUG['click_value_cleared'] += 1
        value = ""

    return {'op': op, 'target_id': str(target_id), 'value': value}


# ─────────────────────────────────────────────
# Rule-based Fallback
# ─────────────────────────────────────────────

def _infer_op_from_task(task: str) -> str:
    task_lower = str(task).lower()
    if re.search(r'\btype\b|\benter\b|\binput\b|fill(?:\s+in)?\b|write\b', task_lower):
        return "TYPE"
    if re.search(r'\bselect\b|\bchoose\b|\bpick\b|\bset\b', task_lower):
        return "SELECT"
    if re.search(r'\bclick\b|\bpress\b|\bsubmit\b|\bopen\b|\bgo\b', task_lower):
        return "CLICK"
    return "CLICK"


def _candidate_relevance_score(candidate: dict[str, Any], task: str, op: str) -> float:
    tag   = str(candidate.get('tag', '')).lower()
    attrs = str(candidate.get('attrs', ''))
    score = _candidate_match_score(candidate, task)

    if op == "TYPE":
        score += 2.0 if tag in {'input', 'textarea'} else -1.0
    elif op == "SELECT":
        score += 2.0 if tag == 'select' else -1.0
        options    = _parse_options(attrs)
        task_words = _token_set(task)
        for opt in options:
            opt_words = _token_set(opt)
            if opt.lower() in str(task).lower():
                score += 2.0
            elif opt_words:
                score += len(opt_words & task_words) / max(len(opt_words), 1)
    elif op == "CLICK":
        score += 1.0 if tag in {'button', 'a'} else 0.0

    return score


def fallback_rule_based(row, candidates):
    """LLM 실패 시 rule 기반으로 op/target_id/value를 결정한다."""
    task      = str(row['task'])
    op        = _infer_op_from_task(task)
    target_id = candidates[0].get('candidate_id', '') if candidates else ""
    value     = ""

    def best_candidate(pool):
        if not pool:
            return None
        return max(
            pool,
            key=lambda c: (_candidate_relevance_score(c, task, op), str(c.get('candidate_id', '')))
        )

    if op == "TYPE":
        pool = [c for c in candidates if str(c.get('tag', '')).lower() in {'input', 'textarea'}]
    elif op == "SELECT":
        pool = [c for c in candidates if str(c.get('tag', '')).lower() == 'select']
    elif op == "CLICK":
        pool = [c for c in candidates if str(c.get('tag', '')).lower() in {'button', 'a'}]
    else:
        pool = []

    chosen = best_candidate(pool) or best_candidate(candidates)
    if chosen:
        target_id = chosen.get('candidate_id', '')
        value     = extract_value_from_task(task, op, str(chosen.get('attrs', '')))

    if op == "CLICK":
        value = ""

    return {'op': op, 'target_id': target_id, 'value': value}


# ─────────────────────────────────────────────
# HTML 타입 감지 및 Workflow 컨텍스트 파싱
# ─────────────────────────────────────────────

def is_workflow_html(html_str) -> bool:
    """Workflow형 HTML 여부 판별."""
    if not isinstance(html_str, str):
        return False
    return ("workflow-context" in html_str) or ("completed-fields" in html_str)


def detect_html_type(row) -> str:
    """행의 cleaned_html을 보고 'workflow' 또는 'real_web' 반환."""
    return "workflow" if is_workflow_html(str(row.get("cleaned_html", ""))) else "real_web"


def extract_workflow_context(html_str) -> dict:
    """Workflow HTML에서 현재 단계 및 완료된 필드 정보를 추출한다.

    반환 예시:
        {"current_step": "6", "total_steps": "7", "completed_fields": ["Name", "Date"]}
    """
    result = {"current_step": "", "total_steps": "", "completed_fields": []}
    if not isinstance(html_str, str):
        return result
    step_m = re.search(r"current step\s+(\d+)\s+of\s+(\d+)", html_str, re.IGNORECASE)
    if step_m:
        result["current_step"] = step_m.group(1)
        result["total_steps"]  = step_m.group(2)
    completed_m = re.search(r"Completed:\s*([^\n<]+)", html_str, re.IGNORECASE)
    if completed_m:
        fields = [f.strip() for f in completed_m.group(1).split(",") if f.strip()]
        result["completed_fields"] = fields
    return result


# ─────────────────────────────────────────────
# 번호 기반 후보 포맷 및 변환
# ─────────────────────────────────────────────

def format_numbered_candidates(candidates) -> str:
    """후보를 1~N 번호로 포맷팅한다.

    LLM이 긴 ID 문자열을 생성하는 대신 숫자(1~15)만 선택하면 됨.
    hallucination 및 형식 오류 대폭 감소.
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        tag         = c.get("tag", "")
        text        = str(c.get("text", "")).strip()
        attrs       = str(c.get("attrs", "")).strip()
        attrs_short = (attrs[:120] + "...") if len(attrs) > 120 else attrs

        line = f"{i:>2}. [{tag}]"
        if text:
            line += f' "{text}"'
        if attrs_short:
            line += f" | {attrs_short}"
        lines.append(line)
    return "\n".join(lines)


def choice_to_candidate_id(choice, candidates) -> str:
    """번호(1~N)를 실제 candidate_id로 변환한다.

    변환 실패 시 빈 문자열 반환 → 호출자가 fallback 처리.
    """
    try:
        idx = int(choice) - 1  # 1-indexed → 0-indexed
        if 0 <= idx < len(candidates):
            return str(candidates[idx].get("candidate_id", ""))
    except (ValueError, TypeError):
        pass
    return ""


# ─────────────────────────────────────────────
# 임베딩 기반 후보 재정렬 (real_web 전용)
# ─────────────────────────────────────────────

def _candidate_description(c) -> str:
    """후보 요소를 짧은 자연어로 변환한다. 있는 데이터만 사용 (할루시네이션 없음)."""
    tag   = c.get("tag", "")
    text  = str(c.get("text", "")).strip()
    attrs = parse_attrs_str(str(c.get("attrs", "")))
    label = (text
             or attrs.get("aria-label", "")
             or attrs.get("placeholder", "")
             or attrs.get("label", "")
             or attrs.get("name", ""))
    options = attrs.get("options", "")
    desc = f"{tag} {label}".strip()
    if options:
        desc += f" choices: {options[:60]}"
    return desc


_embedding_model = None


def rerank_candidates_by_embedding(task: str, candidates: list) -> list:
    """task 임베딩과의 코사인 유사도로 후보를 내림차순 정렬한다 (real_web 전용)."""
    if not candidates:
        return candidates
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    from sentence_transformers import util
    descs     = [_candidate_description(c) for c in candidates]
    task_emb  = _embedding_model.encode([task], convert_to_tensor=True)
    cand_embs = _embedding_model.encode(descs,  convert_to_tensor=True)
    scores    = util.cos_sim(task_emb, cand_embs)[0]
    ranked    = scores.argsort(descending=True).tolist()
    return [candidates[i] for i in ranked]


# ─────────────────────────────────────────────
# RAG 포맷팅 & 프롬프트 빌더
# (train.py / inference.py 공용 — 이 파일만 수정하면 됨)
# ─────────────────────────────────────────────
RETRIEVAL_K             = 3
RETRIEVAL_TASK_CHARS    = 150
RETRIEVAL_HISTORY_CHARS = 150


def format_similar_examples(examples, max_task_chars=None, max_history_chars=None):
    if not examples:
        return "[Similar Past Examples]\nNone"

    def _trunc(s, n):
        if n is None or s is None:
            return s
        s = str(s)
        return s if len(s) <= n else s[:n].rstrip() + "..."

    lines = ["[Similar Past Examples]"]
    for i, ex in enumerate(examples, 1):
        lines.extend([
            f"Example {i}:",
            f"  Task: {_trunc(ex.get('task', ''), max_task_chars)}",
            f"  History: {_trunc(ex.get('history', ''), max_history_chars)}",
            f"  Action: op={ex.get('target_op', '')}, label=\"{ex.get('target_label', '')}\", value=\"{ex.get('target_value', '')}\"",
        ])
    return "\n".join(lines)


def build_prompt(row, candidates, retriever=None, k=RETRIEVAL_K, exclude_id=None):
    """HTML 타입(workflow / real_web)에 따라 최적화된 프롬프트를 생성한다."""
    html_type    = detect_html_type(row)
    n            = len(candidates)
    numbered     = format_numbered_candidates(candidates)
    task_str     = str(row.get("task", ""))
    history_str  = str(row.get("history", "")) or "None"

    if html_type == "workflow":
        ctx = extract_workflow_context(str(row.get("cleaned_html", "")))
        step_hint = ""
        if ctx["current_step"]:
            step_hint = f"Progress: step {ctx['current_step']} of {ctx['total_steps']}"
        completed_hint = ""
        if ctx["completed_fields"]:
            completed_hint = (
                "Already completed (do NOT select these): "
                + ", ".join(ctx["completed_fields"])
            )

        return f"""You are a web UI automation expert filling out a workflow form.
Select the single best action from the numbered candidates below.

[Candidate Elements] — choose one number (1-{n})
{numbered}

[Task]
{task_str}

[History]
{history_str}

[Workflow Status]
{step_hint}
{completed_hint}

Output ONLY valid JSON:
{{"op": "CLICK|TYPE|SELECT", "choice": <number 1-{n}>, "value": "text to type or select, empty string for CLICK"}}"""

    else:  # real_web
        examples_text = ""
        if retriever:
            examples = retriever.query(row, k=k, exclude_id=exclude_id)
            if examples:
                examples_text = format_similar_examples(
                    examples[:k],
                    max_task_chars=RETRIEVAL_TASK_CHARS,
                    max_history_chars=RETRIEVAL_HISTORY_CHARS,
                )

        return f"""You are a web UI automation expert controlling a real website.
Select the single best action from the numbered candidates below.
Candidates are ordered by relevance to the task — earlier numbers are more likely correct.

[Candidate Elements] — choose one number (1-{n})
{numbered}

[Task]
{task_str}

[History]
{history_str}

{examples_text}

Output ONLY valid JSON:
{{"op": "CLICK|TYPE|SELECT", "choice": <number 1-{n}>, "value": "text to type or select, empty string for CLICK"}}"""
