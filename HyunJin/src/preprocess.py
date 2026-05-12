# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
import random
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
    1. SELECT: options= 중 task에 등장하는 옵션 (exact → fuzzy, 동적 임계값)
    2. 따옴표 패턴
    3. TYPE: label/placeholder 기반 뒤따르는 청크 → task 말미 청크
    4. 날짜 regex (date 필드로 확인된 경우에만)
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
                n_words = len(opt_words)
                score = len(opt_words & task_words) / max(n_words, 1)
                if score > best_score:
                    best_score, best_opt = score, opt
            # 동적 임계값: 단어 1개짜리 옵션은 0.3, 3개 이상은 0.6
            if best_opt:
                n = len(_token_set(best_opt))
                threshold = 0.3 if n <= 1 else (0.6 if n >= 3 else 0.5)
                if best_score >= threshold:
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

        # 날짜 regex: date 타입 필드로 확인된 경우에만 반환 (다중 포맷 지원)
        _DATE_PATTERNS = [
            r'\d{4}-\d{2}-\d{2}',                      # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',                  # MM/DD/YYYY or M/D/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',                  # MM-DD-YYYY
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
        ]
        attrs_dict = parse_attrs_str(attrs_str)
        field_type = attrs_dict.get('type', '').lower()
        field_label_all = ' '.join([
            attrs_dict.get('label', ''), attrs_dict.get('placeholder', ''),
            attrs_dict.get('aria-label', ''), attrs_dict.get('name', ''),
        ]).lower()
        if 'date' in field_type or 'date' in field_label_all:
            for _pat in _DATE_PATTERNS:
                date_match = re.search(_pat, task_str, re.IGNORECASE)
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
        # SeeAct empty-text fallback chain
        label = str(
            attrs_dict.get('aria-label', '') or
            attrs_dict.get('placeholder', '') or
            attrs_dict.get('title', '') or
            attrs_dict.get('role', '') or
            attrs_dict.get('label', '') or
            attrs_dict.get('name', '')
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
        # fallback 결과도 cand_map에 없으면 첫 번째 후보로 안전하게 대체
        if target_id not in cand_map and candidates:
            target_id = str(candidates[0].get('candidate_id', ''))
            CONSISTENCY_DEBUG['fallback_target_defaulted'] += 1

    chosen = cand_map.get(str(target_id))
    tag    = str(chosen.get('tag', '')).lower() if chosen else ''

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
        if not fixed_value:
            # options 파싱 실패 시 LLM 원본값 유지, 없으면 task에서 직접 추출
            fixed_value = value or extract_value_from_task(task, 'SELECT', str(chosen.get('attrs', '')) if chosen else '')
            if fixed_value:
                CONSISTENCY_DEBUG['select_value_kept_raw'] += 1
            elif options:
                # 모든 방법 실패 시 첫 번째 옵션 사용 (빈 value 방지)
                fixed_value = options[0]
                CONSISTENCY_DEBUG['select_value_defaulted_first_option'] += 1
        elif fixed_value != value:
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
# HTML 컨텍스트 추출 (real_web 전용)
# ─────────────────────────────────────────────

def _find_element_in_soup(soup, candidate):
    tag   = candidate.get("tag", "") or ""
    attrs = parse_attrs_str(str(candidate.get("attrs", "")))
    text  = str(candidate.get("text", "")).strip()
    for key in ("id", "name", "placeholder", "aria-label"):
        val = attrs.get(key)
        if not val:
            continue
        el = soup.find(tag, attrs={key: val}) if tag else soup.find(attrs={key: val})
        if el:
            return el
    if text and tag:
        for el in soup.find_all(tag):
            if el.get_text(strip=True)[:60] == text[:60]:
                return el
    return None


def get_html_context(soup, candidate) -> str:
    if soup is None:
        return ""
    try:
        el = _find_element_in_soup(soup, candidate)
        if not el:
            return ""
        parts = []
        el_id = el.get("id")
        if el_id:
            label = soup.find("label", {"for": el_id})
            if label:
                t = label.get_text(strip=True)[:50]
                if t:
                    parts.append(f"label:{t}")
        if not parts:
            lby = el.get("aria-labelledby")
            if lby:
                ref = soup.find(id=lby)
                if ref:
                    t = ref.get_text(strip=True)[:50]
                    if t:
                        parts.append(f"label:{t}")
        if not parts:
            prev = el.find_previous_sibling()
            if prev:
                t = prev.get_text(strip=True)[:40]
                if t:
                    parts.append(f"prev:{t}")
        parent = el.parent
        if parent and parent.name not in ("html", "body", "[document]", None):
            p_role = parent.get("aria-label") or parent.get("role") or ""
            p_tag  = parent.name
            if p_tag in ("form", "fieldset", "section", "nav", "header", "main", "footer"):
                parts.append(f"in:<{p_tag}{'[' + p_role[:25] + ']' if p_role else ''}>")

        # 자식 요소 (최대 3개 — 버튼 내부 텍스트, select 옵션 등)
        children = el.find_all(recursive=False)[:3]
        child_texts = []
        for ch in children:
            t = ch.get_text(strip=True)[:25]
            if t:
                child_texts.append(f"{ch.name}:{t}")
        if child_texts:
            parts.append(f"children:[{', '.join(child_texts)}]")

        # DOM 형제 이웃 (최대 5개) — Dual-View +11.9%p 근사
        if parent and parent.name not in ("html", "body", "[document]", None):
            siblings = [s for s in parent.find_all(recursive=False)
                        if s != el and s.get_text(strip=True)][:5]
            nei_texts = [f"{s.name}:{s.get_text(strip=True)[:25]}" for s in siblings]
            if nei_texts:
                parts.append(f"near:[{', '.join(nei_texts)}]")

        return " | ".join(parts)
    except Exception:
        return ""


# ─────────────────────────────────────────────
# 번호 기반 후보 포맷 및 변환
# ─────────────────────────────────────────────

def format_numbered_candidates(candidates, soup=None, compact: bool = False) -> str:
    """후보를 1~N 번호로 포맷팅한다.

    LLM이 긴 ID 문자열을 생성하는 대신 숫자(1~15)만 선택하면 됨.
    핵심 attr(type, name, label, placeholder, aria-label, options)을 앞에 노출하고
    class/data-*/style 등 노이즈 attr은 제거해 모델 구분력을 높인다.
    """
    # AgentOccam KEEP: id/name/type/role/aria-*/placeholder/value/href/for/required/disabled
    _KEY_ATTRS = ("id", "name", "type", "role", "aria-label", "aria-labelledby",
                  "aria-describedby", "aria-expanded", "aria-selected", "aria-checked",
                  "placeholder", "value", "href", "for",
                  "required", "disabled", "label", "options")
    _KEY_ATTRS_COMPACT = ("type", "name", "aria-label", "placeholder", "label", "options", "role")
    # AgentOccam REMOVE: class/style/data-react*/data-v-*/data-test*/xpath/jsname/jscontroller/tabindex
    _NOISE_PREFIXES = ("class", "style", "data-react", "data-v-", "data-test",
                       "xpath", "jsname", "jsaction", "jscontroller", "data-analytics",
                       "tabindex", "data_", "data-ga", "data-track")

    lines = []
    for i, c in enumerate(candidates, 1):
        tag  = c.get("tag", "")
        text = str(c.get("text", "")).strip()
        attrs_dict = parse_attrs_str(str(c.get("attrs", "")))

        # SeeAct empty-text fallback: text 없으면 attrs에서 식별 텍스트 보충
        if not text:
            for _fb_key in ("aria-label", "placeholder", "title", "role", "name"):
                _fb_val = attrs_dict.get(_fb_key, "")
                if _fb_val:
                    text = f"[{_fb_key}:{_fb_val[:40]}]"
                    break

        # 핵심 attr 우선 수집
        key_parts = []
        key_list = _KEY_ATTRS_COMPACT if compact else _KEY_ATTRS
        for k in key_list:
            v = attrs_dict.get(k, "")
            if v:
                key_parts.append(f"{k}={v[:60]}")

        # 핵심 attr에 없는 나머지 중 노이즈 아닌 것 추가 (최대 2개)
        if not compact:
            extras = []
            for k, v in attrs_dict.items():
                if k in _KEY_ATTRS:
                    continue
                if any(k.startswith(p) for p in _NOISE_PREFIXES):
                    continue
                extras.append(f"{k}={str(v)[:40]}")
                if len(extras) >= 2:
                    break
            key_parts.extend(extras)

        line = f"{i:>2}. [{tag}]"
        if text:
            line += f' "{text[:60]}"'
        if key_parts:
            line += " | " + " | ".join(key_parts)

        # HTML 컨텍스트 (soup가 있을 때만)
        ctx = ""
        if not compact and soup is not None:
            ctx = get_html_context(soup, c)
        if ctx:
            line += f"\n    ~ {ctx}"

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


_rerank_model = None


def rerank_candidates_by_embedding(task: str, candidates: list) -> list:
    """task와 각 후보를 cross-encoder로 공동 인코딩해 내림차순 정렬한다 (real_web 전용)."""
    if not candidates:
        return candidates
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    descs  = [_candidate_description(c) for c in candidates]
    scores = _rerank_model.predict([(task, d) for d in descs])
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in ranked]


def hard_negative_shuffle(candidates: list, target_id: str) -> list:
    """셔플 후 정답과 같은 tag의 후보(hard negative)를 정답 바로 앞에 배치한다.

    같은 tag 후보가 없으면 일반 random.shuffle과 동일하게 동작한다.
    """
    shuffled = candidates.copy()
    random.shuffle(shuffled)

    target_pos = next(
        (i for i, c in enumerate(shuffled) if str(c.get("candidate_id", "")) == target_id),
        None,
    )
    if target_pos is None:
        return shuffled

    target_tag = shuffled[target_pos].get("tag", "")
    hn_pos = next(
        (i for i, c in enumerate(shuffled)
         if i != target_pos and c.get("tag", "") == target_tag),
        None,
    )
    # hard negative를 정답 바로 앞으로 swap (정답이 맨 앞이면 뒤로)
    if hn_pos is not None:
        swap_to = (target_pos - 1) if target_pos > 0 else (target_pos + 1)
        if swap_to < len(shuffled):
            shuffled[hn_pos], shuffled[swap_to] = shuffled[swap_to], shuffled[hn_pos]

    return shuffled


def generate_cot_reasoning(task: str, op: str, choice: int, candidate: dict, value: str,
                           candidates: list = None) -> str:
    """학습 데이터용: task-aware 추론 문장 생성 (CoT SFT).

    candidates가 주어지면 정답이 아닌 후보를 간략히 언급해 비교 추론을 학습시킨다.
    전체 출력은 160자 이하 단일 문자열(줄바꿈 없음)로 유지한다.
    """
    tag   = candidate.get("tag", "")
    text  = str(candidate.get("text", "")).strip()
    attrs = parse_attrs_str(str(candidate.get("attrs", "")))
    label = (text
             or attrs.get("aria-label", "")
             or attrs.get("placeholder", "")
             or attrs.get("label", "")
             or attrs.get("name", "")).strip()

    task_short = str(task).strip()[:50]
    el = f"El.{choice}[{tag}" + (f":{label[:25]}" if label else "") + "]"

    # 비교 대상: 정답과 tag가 다른 후보 최대 2개 (token 절약)
    contrast = ""
    if candidates:
        others = [
            f"el.{i}[{c.get('tag','')}]"
            for i, c in enumerate(candidates, 1)
            if i != choice and str(c.get("tag", "")) != tag
        ][:2]
        if others:
            contrast = f" (not {', '.join(others)})"

    if op == "CLICK":
        base = f"Need CLICK. {el} matches '{task_short}'.{contrast}"
    elif op == "TYPE":
        base = f"Need TYPE '{value[:25]}'. {el} is the right field for '{task_short}'.{contrast}"
    else:  # SELECT
        base = f"Need SELECT '{value[:25]}'. {el} has matching option for '{task_short}'.{contrast}"

    return base[:160]


# ─────────────────────────────────────────────
# History 압축 (AgentOccam: 피봇 step만 남김)
# ─────────────────────────────────────────────

def _compress_history(history_str: str) -> str:
    """AgentOccam: tag+text+op만 남겨 60-70% 토큰 절약.

    입력: "Step 1: [button] Submit -> CLICK\nStep 2: [input] Email -> TYPE: foo@bar.com"
    출력: "[button]Submit→CLICK | [input]Email→TYPE:foo@bar.com"
    """
    if not history_str or str(history_str).strip() in ("", "None", "nan"):
        return "None"
    steps = re.findall(
        r'Step\s+\d+:\s*\[([^\]]+)\]\s*(.*?)\s*->\s*(\w+)(?::\s*([^\n]*))?',
        str(history_str)
    )
    if not steps:
        return str(history_str)[:300]
    compressed = []
    for tag, text, op, value in steps:
        text = text.strip()[:30]
        entry = f"[{tag}]{' ' + text if text else ''}→{op}"
        if value and op != "CLICK":
            entry += f":{value.strip()[:20]}"
        compressed.append(entry)
    return " | ".join(compressed)


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


def build_prompt(
    row,
    candidates,
    retriever=None,
    k=RETRIEVAL_K,
    exclude_id=None,
    action_desc: str = "",
    compact_candidates: bool = False,
):
    """HTML 타입(workflow / real_web)에 따라 최적화된 프롬프트를 생성한다."""
    from bs4 import BeautifulSoup

    html_type   = detect_html_type(row)
    html_str    = str(row.get("cleaned_html", ""))
    task_str    = str(row.get("task", ""))
    history_str = _compress_history(str(row.get("history", "")))
    n           = len(candidates)

    # real_web: HTML 파싱 후 컨텍스트 추출
    soup = None
    if html_type == "real_web" and html_str and html_str != "nan":
        try:
            soup = BeautifulSoup(html_str, "lxml")
        except Exception:
            soup = None

    numbered = format_numbered_candidates(candidates, soup=soup, compact=compact_candidates)

    if html_type == "workflow":
        ctx = extract_workflow_context(html_str)
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
        plan_block = f"\n\n[Action Plan]\n{action_desc}" if action_desc else ""
        return f"""You are a web UI automation expert controlling a real website.
Select the single best action from the numbered candidates below.

[Candidate Elements] — choose one number (1-{n})
{numbered}

[Task]
{task_str}{plan_block}

[History]
{history_str}

Output ONLY valid JSON:
{{"op": "CLICK|TYPE|SELECT", "choice": <number 1-{n}>, "value": "text to type or select, empty string for CLICK"}}"""


def build_grounding_step1_prompt(row) -> str:
    """2단계 Grounding Step 1: 후보 없이 task+history만으로 다음 액션을 자연어 기술.

    출력: {"op": "CLICK|TYPE|SELECT", "action_desc": "어떤 요소에 어떤 동작을 해야 하는지"}
    """
    task_str    = str(row.get("task", ""))
    history_str = _compress_history(str(row.get("history", "")))

    return f"""You are a web UI automation expert.
Based ONLY on the task and history, describe what action needs to be performed next.
Do NOT choose a specific element yet — just describe what kind of element and what to do.

[Task]
{task_str}

[History]
{history_str}

Output ONLY valid JSON:
{{"op": "CLICK|TYPE|SELECT", "action_desc": "brief description of what element to interact with and what to do"}}"""


def generate_action_desc(op: str, target_cand: dict, val: str) -> str:
    """학습 데이터용: 정답 기반으로 Step 1의 action_desc 문장 자동 생성."""
    tag   = target_cand.get("tag", "element")
    text  = str(target_cand.get("text", "")).strip()
    attrs = parse_attrs_str(str(target_cand.get("attrs", "")))
    label = (text
             or attrs.get("aria-label", "")
             or attrs.get("placeholder", "")
             or attrs.get("label", "")
             or attrs.get("name", "")).strip()
    label_part = f" '{label[:40]}'" if label else ""
    if op == "CLICK":
        return f"Click the {tag}{label_part}"
    elif op == "TYPE":
        return f"Type '{val[:30]}' into the {tag}{label_part} field"
    else:
        return f"Select '{val[:30]}' from the {tag}{label_part}"


def build_group_prompt(
    row,
    group_candidates,
    soup=None,
    action_desc: str = "",
    compact_candidates: bool = False,
) -> str:
    """토너먼트용: 5개 이하 후보 중 정답을 고르거나 None(choice=0) 반환.

    action_desc: Step 1 Grounding에서 생성된 자연어 액션 기술 (있으면 주입)
    """
    n           = len(group_candidates)
    numbered    = format_numbered_candidates(group_candidates, soup=soup, compact=compact_candidates)
    task_str    = str(row.get("task", ""))
    history_str = _compress_history(str(row.get("history", "")))
    plan_block  = f"\n[Action Plan]\n{action_desc}" if action_desc else ""

    return f"""You are a web UI automation expert.
Does ANY of these {n} elements match the task? Choose 1-{n}, or 0 if none match.

[Candidates] — choose one number (1-{n}) or 0 for none:
{numbered}

[Task]
{task_str}{plan_block}

[History]
{history_str}

Output ONLY valid JSON:
{{"choice": <1-{n} or 0>, "op": "CLICK|TYPE|SELECT", "value": "text to type or select, empty string for CLICK"}}"""
