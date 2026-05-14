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
    matches = _find_elements_in_soup(soup, candidate)
    return matches[0] if matches else None


def _norm_match_text(value) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip().lower()


def _norm_attr_key(key: str) -> str:
    return str(key or "").strip().lower().replace("_", "-")


def _canonical_attr_items(attrs_str: str) -> tuple:
    attrs = parse_attrs_str(str(attrs_str or ""))
    return tuple(
        sorted(
            (_norm_attr_key(k), _norm_match_text(v))
            for k, v in attrs.items()
            if _norm_match_text(v)
        )
    )


def _candidate_signature(candidate) -> tuple:
    """Stable row-local signature used to detect indistinguishable candidates."""
    return (
        _norm_match_text(candidate.get("tag", "")),
        _norm_match_text(candidate.get("text", "")),
        _canonical_attr_items(str(candidate.get("attrs", ""))),
    )


def _html_attr(el, key: str) -> str:
    """Read an HTML attribute while tolerating aria-label vs aria_label variants."""
    if not el:
        return ""
    keys = [key, key.replace("-", "_"), key.replace("_", "-")]
    for k in keys:
        if el.has_attr(k):
            v = el.get(k)
            if isinstance(v, list):
                v = " ".join(str(x) for x in v)
            return str(v)
    return ""


def _find_elements_in_soup(soup, candidate, limit: int = 80) -> list:
    """Best-effort candidate-to-DOM matching.

    candidate_id is not linked to backend_node_id, so this relies only on visible
    candidate fields. Blank tag/text/attrs candidates intentionally return no
    match instead of guessing an arbitrary DOM node.
    """
    if soup is None:
        return []
    tag = str(candidate.get("tag", "") or "").strip()
    text = str(candidate.get("text", "") or "").strip()
    attrs = parse_attrs_str(str(candidate.get("attrs", "")))
    useful_attrs = {
        _norm_attr_key(k): str(v)
        for k, v in attrs.items()
        if _norm_match_text(v)
        and _norm_attr_key(k) not in {"class", "style", "data-reactid"}
    }
    if not tag and not text and not useful_attrs:
        return []

    pool = soup.find_all(tag) if tag else soup.find_all(True)
    scored = []
    text_norm = _norm_match_text(text)

    for order, el in enumerate(pool):
        score = 0
        if text_norm:
            el_text = _norm_match_text(el.get_text(" ", strip=True))
            if el_text == text_norm:
                score += 8
            elif el_text.startswith(text_norm) or text_norm in el_text:
                score += 4

        for k, v in useful_attrs.items():
            actual = _norm_match_text(_html_attr(el, k))
            expected = _norm_match_text(v)
            if actual and actual == expected:
                score += 6 if k in {"id", "name", "aria-label", "placeholder", "label"} else 3

        if score > 0:
            scored.append((score, order, el))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [el for _, _, el in scored[:limit]]


def _summarize_dom_context(el) -> str:
    """Compact parent/sibling context for disambiguating same-looking candidates."""
    if not el:
        return ""
    parts = []
    parent = el.parent
    if parent and parent.name not in ("html", "body", "[document]", None):
        p_attrs = []
        for key in ("id", "role", "aria-label", "class"):
            val = _html_attr(parent, key)
            if val:
                p_attrs.append(f"{key}={val[:28]}")
        p_text = parent.get_text(" ", strip=True)
        if p_text:
            p_attrs.append(f"text={p_text[:45]}")
        parts.append(f"parent:<{parent.name}{' ' + ' '.join(p_attrs) if p_attrs else ''}>")

    prev = el.find_previous_sibling()
    if prev:
        t = prev.get_text(" ", strip=True)
        if t:
            parts.append(f"prev:{prev.name}:{t[:32]}")
    nxt = el.find_next_sibling()
    if nxt:
        t = nxt.get_text(" ", strip=True)
        if t:
            parts.append(f"next:{nxt.name}:{t[:32]}")

    own_text = el.get_text(" ", strip=True)
    if own_text:
        parts.append(f"self_text:{own_text[:40]}")

    return " ; ".join(parts[:4])


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
# 계층적 A-Tree 포맷팅 (Hierarchy-aware)
# ─────────────────────────────────────────────

def _get_ancestor_path(soup, el) -> list:
    """요소의 조상 노드 리스트를 반환한다 (Root -> Parent)."""
    if not el:
        return []
    path = []
    curr = el.parent
    while curr and curr.name not in ("html", "body", "[document]", None):
        path.append(curr)
        curr = curr.parent
    return path[::-1]


def _format_node_minimal(el) -> str:
    """조상 노드를 위한 최소 정보 포맷 (tag, id, class, role)."""
    tag = el.name
    attrs = el.attrs
    parts = [f"<{tag}"]
    if "id" in attrs:
        parts.append(f"id={attrs['id']}")
    if "class" in attrs:
        cls = " ".join(attrs["class"]) if isinstance(attrs["class"], list) else str(attrs["class"])
        parts.append(f"class={cls[:30]}")
    role = attrs.get("role") or attrs.get("aria-label")
    if role:
        parts.append(f"role={str(role)[:30]}")
    return " ".join(parts) + ">"


def format_as_tree(candidates, soup=None, compact: bool = False) -> str:
    """후보 요소와 그 조상들을 포함하는 인덴트 기반 요약 트리를 생성한다.

    1. 모든 후보의 조상들을 '필수 노드'로 식별.
    2. DFS 탐색을 통해 필수 노드만 출력.
    3. 후보 노드는 번호[1]와 전체 속성을, 조상 노드는 최소 정보만 표시.
    """
    if soup is None:
        return format_numbered_candidates(candidates, soup=None, compact=compact)

    # 1. 필수 노드 매핑 (candidate -> soup_element)
    cand_to_el = {}
    relevant_nodes = set()
    for i, c in enumerate(candidates, 1):
        el = _find_element_in_soup(soup, c)
        if el:
            cand_to_el[i] = el
            relevant_nodes.add(el)
            for anc in _get_ancestor_path(soup, el):
                relevant_nodes.add(anc)

    if not cand_to_el:
        return format_numbered_candidates(candidates, soup=None, compact=compact)

    # 2. DFS 트리 생성
    _KEY_ATTRS_COMPACT = ("type", "name", "aria-label", "placeholder", "label", "options", "role")
    _KEY_ATTRS = ("id", "name", "type", "role", "aria-label", "placeholder", "value", "href", "for", "label", "options")

    lines = []
    
    def _render(node, depth):
        is_candidate = False
        cand_idx = None
        for idx, el in cand_to_el.items():
            if el == node:
                is_candidate = True
                cand_idx = idx
                break
        
        indent = "  " * depth
        if is_candidate:
            c = candidates[cand_idx - 1]
            tag = c.get("tag", "")
            text = str(c.get("text", "")).strip()
            attrs_dict = parse_attrs_str(str(c.get("attrs", "")))
            
            # SeeAct fallback text
            if not text:
                for fb in ("aria-label", "placeholder", "title", "role", "name"):
                    if attrs_dict.get(fb):
                        text = f"[{fb}:{attrs_dict[fb][:30]}]"
                        break
            
            key_list = _KEY_ATTRS_COMPACT if compact else _KEY_ATTRS
            attr_parts = [f"{k}={str(attrs_dict[k])[:50]}" for k in key_list if attrs_dict.get(k)]
            
            line = f"{indent}[{cand_idx}] <{tag}>"
            if text: line += f" '{text[:50]}'"
            if attr_parts: line += " | " + " | ".join(attr_parts)
            lines.append(line)
        else:
            lines.append(f"{indent}{_format_node_minimal(node)}")

        # 자식 노드 중 relevant_nodes에 속하는 것만 방문
        for child in node.find_all(recursive=False):
            if child in relevant_nodes:
                _render(child, depth + 1)

    # Root 영역들 찾기 (가장 상위의 relevant_nodes)
    roots = []
    for node in relevant_nodes:
        if not any(p in relevant_nodes for p in node.parents):
            roots.append(node)
    
    for r in roots:
        _render(r, 0)
        
    return "\n".join(lines)


def _format_candidate_base_line(i: int, candidate: dict, compact: bool = False) -> str:
    tag = candidate.get("tag", "")
    text = str(candidate.get("text", "")).strip()
    attrs_dict = parse_attrs_str(str(candidate.get("attrs", "")))

    if not text:
        for fb_key in ("aria-label", "aria_label", "placeholder", "title", "role", "name", "label"):
            fb_val = attrs_dict.get(fb_key, "")
            if fb_val:
                text = f"[{fb_key}:{str(fb_val)[:40]}]"
                break

    key_attrs = (
        ("type", "name", "aria-label", "aria_label", "placeholder", "label", "options", "role")
        if compact
        else (
            "id", "name", "type", "role", "aria-label", "aria_label", "aria-labelledby",
            "aria_describedby", "aria-describedby", "placeholder", "value", "href",
            "for", "label", "options",
        )
    )
    key_parts = []
    for key in key_attrs:
        value = attrs_dict.get(key, "")
        if value:
            key_parts.append(f"{key}={str(value)[:60]}")

    line = f"{i:>2}. [{tag}]"
    if text:
        line += f' "{text[:60]}"'
    if key_parts:
        line += " | " + " | ".join(key_parts)
    return line


def format_contextual_candidates(candidates, soup=None, compact: bool = False) -> str:
    """List every candidate and append best-effort DOM context when available."""
    from collections import Counter

    signatures = [_candidate_signature(c) for c in candidates]
    sig_counts = Counter(signatures)
    sig_seen = Counter()
    sig_matches = {}
    lines = []

    for i, candidate in enumerate(candidates, 1):
        sig = signatures[i - 1]
        occurrence = sig_seen[sig]
        sig_seen[sig] += 1

        line = _format_candidate_base_line(i, candidate, compact=compact)
        extras = []
        group_size = sig_counts[sig]
        if group_size > 1:
            extras.append(f"dup={occurrence + 1}/{group_size}")

        if soup is not None:
            if sig not in sig_matches:
                sig_matches[sig] = _find_elements_in_soup(soup, candidate)
            matches = sig_matches[sig]
            chosen_el = None
            if matches:
                if group_size > 1 and len(matches) >= group_size:
                    chosen_el = matches[min(occurrence, len(matches) - 1)]
                    extras.append("dom=occurrence-best-effort")
                elif len(matches) == 1:
                    chosen_el = matches[0]
                    extras.append("dom=single")
                else:
                    chosen_el = matches[0]
                    extras.append(f"dom=ambiguous-top1/{len(matches)}")

                ctx = _summarize_dom_context(chosen_el)
                if ctx:
                    extras.append(f"ctx={ctx}")
            elif group_size > 1:
                extras.append("dom=no-unique-match")

        if extras:
            line += " | " + " | ".join(extras)
        lines.append(line)

    return "\n".join(lines)


def format_numbered_candidates(candidates, soup=None, compact: bool = False) -> str:
    """후보를 1~N 번호로 포맷팅한다. soup가 있으면 계층적 트리 모드로 동작."""
    if soup is not None:
        try:
            contextual = format_contextual_candidates(candidates, soup, compact)
            if contextual.strip():
                return contextual
        except Exception:
            pass # 실패 시 리스트 모드로 fallback

    # --- 기존 리스트 모드 (List Mode) ---
    _KEY_ATTRS = ("id", "name", "type", "role", "aria-label", "aria-labelledby",
                  "aria-describedby", "aria-expanded", "aria-selected", "aria-checked",
                  "placeholder", "value", "href", "for",
                  "required", "disabled", "label", "options")
    _KEY_ATTRS_COMPACT = ("type", "name", "aria-label", "placeholder", "label", "options", "role")
    _NOISE_PREFIXES = ("class", "style", "data-react", "data-v-", "data-test",
                       "xpath", "jsname", "jsaction", "jscontroller", "data-analytics",
                       "tabindex", "data_", "data-ga", "data-track")

    lines = []
    for i, c in enumerate(candidates, 1):
        tag  = c.get("tag", "")
        text = str(c.get("text", "")).strip()
        attrs_dict = parse_attrs_str(str(c.get("attrs", "")))

        if not text:
            for _fb_key in ("aria-label", "placeholder", "title", "role", "name"):
                _fb_val = attrs_dict.get(_fb_key, "")
                if _fb_val:
                    text = f"[{_fb_key}:{_fb_val[:40]}]"
                    break

        key_parts = []
        key_list = _KEY_ATTRS_COMPACT if compact else _KEY_ATTRS
        for k in key_list:
            v = attrs_dict.get(k, "")
            if v:
                key_parts.append(f"{k}={v[:60]}")

        if not compact:
            extras = []
            for k, v in attrs_dict.items():
                if k in _KEY_ATTRS or any(k.startswith(p) for p in _NOISE_PREFIXES):
                    continue
                extras.append(f"{k}={str(v)[:40]}")
                if len(extras) >= 2: break
            key_parts.extend(extras)

        line = f"{i:>2}. [{tag}]"
        if text: line += f' "{text[:60]}"'
        if key_parts: line += " | " + " | ".join(key_parts)
        lines.append(line)
    return "\n".join(lines)


def choice_to_candidate_id(choice, candidates) -> str:
    """번호(1~N)를 실제 candidate_id로 변환한다."""
    try:
        idx = int(choice) - 1
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


def rerank_candidates_by_embedding(task: str, candidates: list, use_rerank: bool = True) -> list:
    """task와 각 후보를 cross-encoder로 공동 인코딩해 내림차순 정렬한다 (전체 후보 정렬)."""
    if not use_rerank or not candidates:
        return candidates
    global _rerank_model
    if _rerank_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            return candidates
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
                           candidates: list = None, soup=None) -> str:
    """학습 데이터용: task-aware + 구조 기반 추론 문장 생성.

    A-Tree 구조 정보를 반영하여 '어떤 섹션(parent)에 있는지'를 CoT에 포함하고,
    유사한 오답(sibling)과의 차별점을 강조하여 grounding 능력을 키운다.
    """
    tag   = candidate.get("tag", "")
    text  = str(candidate.get("text", "")).strip()
    attrs = parse_attrs_str(str(candidate.get("attrs", "")))
    label = (text or attrs.get("aria-label", "") or attrs.get("placeholder", "") or 
             attrs.get("label", "") or attrs.get("name", "")).strip()

    # 1. 구조적 경로 파악 (Hierarchy path)
    ctx_path = []
    if soup:
        el = _find_element_in_soup(soup, candidate)
        if el:
            curr = el.parent
            while curr and curr.name not in ("html", "body", "[document]", None):
                p_tag = curr.name
                # id, aria-label, role 순으로 힌트 추출
                p_hint = curr.get("id") or curr.get("aria-label") or curr.get("role") or ""
                ctx_path.append(f"{p_tag}{'#' + str(p_hint)[:15] if p_hint else ''}")
                curr = curr.parent
                if len(ctx_path) >= 2: break # 상위 2단계까지만 포함하여 간결성 유지
    
    path_str = " > ".join(reversed(ctx_path))
    ctx_hint = f" located under {path_str}," if path_str else ""

    task_clean = str(task).strip()[:40]
    el_desc = f"element [{choice}] <{tag}> '{label[:20]}'"

    # 2. 비교 추론 (Sibling/Similar tag 차별화)
    contrast = ""
    if candidates:
        similars = [i for i, c in enumerate(candidates, 1) 
                    if i != choice and str(c.get("tag", "")) == tag][:1]
        if similars:
            contrast = f" (choosing this over El.{similars[0]} due to closer task relevance)"

    if op == "CLICK":
        reason = f"To '{task_clean}',{ctx_hint} I need to CLICK {el_desc}{contrast}."
    elif op == "TYPE":
        reason = f"For '{task_clean}',{ctx_hint} I will TYPE '{value[:20]}' into {el_desc}{contrast}."
    else:  # SELECT
        reason = f"To fulfill '{task_clean}',{ctx_hint} I must SELECT '{value[:20]}' using {el_desc}{contrast}."

    return reason[:200]


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
# LCA (Lowest Common Ancestor) 분석 (DPO용)
# ─────────────────────────────────────────────

def get_dom_distance(soup, el1, el2) -> int:
    """두 BeautifulSoup 요소 간의 DOM 트리 거리를 계산한다.
    거리 = depth(el1) + depth(el2) - 2 * depth(LCA(el1, el2))
    """
    if not el1 or not el2:
        return 999  # 거리를 측정할 수 없는 경우
    if el1 == el2:
        return 0

    # 각 요소의 조상 노드 리스트 (root 방향)
    path1 = [el1] + list(el1.parents)
    path2 = [el2] + list(el2.parents)

    # Root부터 내려오면서 공통 조상이 끝나는 지점 찾기
    path1.reverse()
    path2.reverse()

    common_depth = 0
    for n1, n2 in zip(path1, path2):
        if n1 == n2:
            common_depth += 1
        else:
            break

    return (len(path1) + len(path2)) - (2 * common_depth)


def _is_false_negative(cand, target_cand):
    """정답(target_cand)과 지나치게 유사한 후보인지 판별한다. (False Negative 필터)"""
    if not cand or not target_cand:
        return True
    if str(cand.get("candidate_id")) == str(target_cand.get("candidate_id")):
        return True

    t1 = str(cand.get("text", "")).strip().lower()
    t2 = str(target_cand.get("text", "")).strip().lower()

    # 1. 강한 제외 조건: text가 비어있지 않고 완전히 동일
    if t1 and t1 == t2:
        return True

    a1 = parse_attrs_str(str(cand.get("attrs", "")))
    a2 = parse_attrs_str(str(target_cand.get("attrs", "")))

    # 2. 강한 제외 조건: label, aria-label 중 하나라도 비어있지 않고 완전히 동일
    for key in ["label", "aria-label"]:
        v1 = str(a1.get(key, "")).strip().lower()
        v2 = str(a2.get(key, "")).strip().lower()
        if v1 and v1 == v2:
            return True

    # 3. 약한 유사성 조건: name, placeholder 일치 여부 확인
    weak_matches = 0
    for key in ["name", "placeholder"]:
        v1 = str(a1.get(key, "")).strip().lower()
        v2 = str(a2.get(key, "")).strip().lower()
        if v1 and v1 == v2:
            weak_matches += 1

    # 4. 약한 조건이 2개 이상 일치하면 제외
    if weak_matches >= 2:
        return True

    return False


def find_lca_hard_negative(row, candidates, target_id):
    """15개 후보 중 정답(target_id)과 가장 헷갈릴만한 '매력적인 오답'을 반환한다.
    
    WEPO+ 파이프라인:
    1. DOM 거리(LCA)가 가까움 (구조적 유사도)
    2. Reranker 점수가 높음 (의미적 유사도)
    3. False-negative 필터 적용
    """
    from bs4 import BeautifulSoup

    # 1. 정답 요소 찾기 및 기본 필터링
    target_cand = next((c for c in candidates if str(c.get("candidate_id", "")) == target_id), None)
    others = [c for c in candidates if str(c.get("candidate_id", "")) != target_id]
    
    if not target_cand or not others:
        return random.choice(others) if others else None

    # False-negative 필터링 (정답과 너무 비슷한 건 오답 샘플로 부적합)
    filtered_others = [c for c in others if not _is_false_negative(c, target_cand)]
    if not filtered_others:
        filtered_others = others

    html_str = str(row.get("cleaned_html", ""))
    soup = None
    if html_str and html_str != "nan":
        try:
            soup = BeautifulSoup(html_str, "lxml")
        except: pass

    target_el = _find_element_in_soup(soup, target_cand) if soup else None

    # 2. 하이브리드 스코어링 (LCA + Reranker)
    # Reranker 점수를 미리 계산하기 위해 candidates 리스트 활용
    task = str(row.get("task", ""))
    reranked_others = rerank_candidates_by_embedding(task, filtered_others)
    
    # Rerank 순위 기반 점수 (앞쪽에 올수록 높은 점수)
    rerank_scores = {c["candidate_id"]: (len(reranked_others) - i) / len(reranked_others) 
                     for i, c in enumerate(reranked_others)}

    best_neg = None
    max_total_score = -999.0

    for c in filtered_others:
        # A. 물리적 점수 (LCA)
        dist_score = 0.0
        sibling_bonus = 0.0
        if soup and target_el:
            c_el = _find_element_in_soup(soup, c)
            if c_el:
                dist = get_dom_distance(soup, target_el, c_el)
                dist_score = max(0, 20 - dist) / 20.0
                if c_el.parent == target_el.parent:
                    sibling_bonus = 0.5 # 같은 부모면 매우 강력한 오답 후보

        # B. 의미적 점수 (Reranker)
        semantic_score = rerank_scores.get(c["candidate_id"], 0.0)
        
        # C. 태그 보너스
        tag_bonus = 0.2 if str(c.get("tag")).lower() == str(target_cand.get("tag")).lower() else 0.0

        total_score = dist_score + sibling_bonus + (semantic_score * 1.5) + tag_bonus

        if total_score > max_total_score:
            max_total_score = total_score
            best_neg = c

    return best_neg or random.choice(filtered_others)


# ─────────────────────────────────────────────
# RAG 포맷팅 & 프롬프트 빌더
# ─────────────────────────────────────────────
RETRIEVAL_K             = 5     # 80GB VRAM: 더 많은 사례(Few-shot) 수용
RETRIEVAL_TASK_CHARS    = 200
RETRIEVAL_HISTORY_CHARS = 200


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
    compact_candidates: bool = False,
    use_rerank: bool = True,
    return_candidates: bool = False,
):
    """Optimized English-only prompt for UI-TARS / Qwen2.5 Agentic Reasoning."""
    from bs4 import BeautifulSoup

    html_type   = detect_html_type(row)
    html_str    = str(row.get("cleaned_html", ""))
    task_str    = str(row.get("task", ""))
    history_str = _compress_history(str(row.get("history", "")))

    # 1. Universal Rerank (Put the most relevant elements at the top)
    candidates = rerank_candidates_by_embedding(task_str, candidates, use_rerank=use_rerank)
    n = len(candidates)

    # 2. RAG (Inject past successful examples)
    rag_block = ""
    if retriever:
        examples = retriever.query(row, k=k, exclude_id=exclude_id)
        rag_block = format_similar_examples(examples, RETRIEVAL_TASK_CHARS, RETRIEVAL_HISTORY_CHARS) + "\n\n"

    # 3. HTML Parsing & A-Tree formatting
    soup = None
    if html_str and html_str != "nan":
        try:
            soup = BeautifulSoup(html_str, "lxml")
        except: soup = None
    numbered = format_numbered_candidates(candidates, soup=soup, compact=compact_candidates)

    # 4. Thinking Guidelines (Force logical grounding in English)
    reasoning_guideline = """[Reasoning Guidelines]
1. Task Goal: Understand the current objective from 'Task' and 'History'.
2. Element Analysis: Look for labels, IDs, ARIA roles, or values in 'Candidate Elements' that match the goal.
3. Contrast: If multiple elements look similar, use duplicate markers and DOM parent/sibling context to distinguish them.
4. Consistency: Ensure the predicted operation (CLICK/TYPE/SELECT) matches the element's tag (e.g., TYPE into input).
5. Output: Provide reasoning in <think> tags, followed by the JSON action."""

    # 5. Core Instruction Assembly
    role_instruction = "You are a specialized Web UI automation agent. Predict the next action to fulfill the task."
    
    prompt = f"""{role_instruction}

{rag_block}[Candidate Elements] (Total: {n})
{numbered}

[Task]
{task_str}

[History]
{history_str}

{reasoning_guideline}

[Output Format]
<think> (Brief step-by-step reasoning in English) </think>
{{"op": "CLICK|TYPE|SELECT", "choice": <1-{n}>, "value": "string"}}"""

    # Add workflow-specific status if applicable
    if html_type == "workflow":
        ctx = extract_workflow_context(html_str)
        workflow_hint = f"\n\n[Workflow Status]\nStep {ctx['current_step']} of {ctx['total_steps']}\n"
        if ctx["completed_fields"]:
            workflow_hint += f"Completed Fields: {', '.join(ctx['completed_fields'])}\n"
        final_prompt = prompt + workflow_hint
        return (final_prompt, candidates) if return_candidates else final_prompt
    
    return (prompt, candidates) if return_candidates else prompt
