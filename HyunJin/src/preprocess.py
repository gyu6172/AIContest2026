import pandas as pd
import json
import re
from collections import Counter
from typing import Any

CONSISTENCY_DEBUG = Counter()

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
    """History의 op 시퀀스를 시그니처로 사용하여 워크플로우 분기(Branch) 구별"""
    if pd.isna(history_str) or not str(history_str).strip():
        return "START"
    ops = re.findall(r'->\s*(CLICK|TYPE|SELECT)', str(history_str))
    if not ops:
        return "STEP_" + str(len(re.findall(r'Step \d+:', str(history_str))) + 1)
    return ",".join(ops)

def build_empirical_priors(train_df):
    """Build weak empirical priors that do not depend on history flow.

    These priors are intended for prompt context / tie-breaking only. They
    deliberately avoid `(site_token, history_signature)` next-step templates.
    """
    priors = {
        'global_op': Counter(),
        'global_tag_op': {},
        'site_tag_op': {},
        'site_labels': {},
    }

    tag_op_counts = {}
    site_tag_op_counts = {}
    site_label_counts = {}

    for _, row in train_df.iterrows():
        op = str(row.get('op', 'CLICK'))
        site = str(row.get('site_token', ''))
        priors['global_op'][op] += 1

        try:
            candidates = json.loads(row.get('candidate_elements', '[]'))
        except Exception:
            candidates = []

        target_id = str(row.get('target_id', ''))
        target = next((c for c in candidates if str(c.get('candidate_id', '')) == target_id), None)
        if not target:
            continue

        tag = str(target.get('tag', '')).lower()
        label = _candidate_label(target)

        tag_op_counts.setdefault(tag, Counter())[op] += 1
        site_tag_op_counts.setdefault(site, {}).setdefault(tag, Counter())[op] += 1
        if label:
            site_label_counts.setdefault(site, Counter())[(op, tag, label)] += 1

    priors['global_op'] = dict(priors['global_op'])
    priors['global_tag_op'] = {
        tag: dict(counts) for tag, counts in tag_op_counts.items()
    }
    priors['site_tag_op'] = {
        site: {tag: dict(counts) for tag, counts in per_tag.items()}
        for site, per_tag in site_tag_op_counts.items()
    }
    priors['site_labels'] = {
        site: [
            {'op': item[0][0], 'tag': item[0][1], 'label': item[0][2], 'count': item[1]}
            for item in counts.most_common(8)
        ]
        for site, counts in site_label_counts.items()
    }
    return priors

def _format_counter_dist(counts, limit=None):
    if not counts:
        return "none"
    items = Counter(counts).most_common(limit)
    total = sum(counts.values()) or 1
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)

def format_empirical_priors(row, priors):
    """Format weak priors for the LLM prompt."""
    if not priors:
        return "[Empirical Priors]\nNone"

    site = str(row.get('site_token', ''))
    lines = [
        "[Empirical Priors]",
        "Task and current candidates are the primary evidence. Use these priors only as weak tie-breakers.",
        f"Global op distribution: {_format_counter_dist(priors.get('global_op', {}))}",
    ]

    global_tag_op = priors.get('global_tag_op', {})
    if global_tag_op:
        tag_lines = []
        for tag, counts in sorted(global_tag_op.items()):
            tag_lines.append(f"{tag or 'unknown'} -> {_format_counter_dist(counts)}")
        lines.append("Global tag/op: " + "; ".join(tag_lines[:8]))

    site_tag_op = priors.get('site_tag_op', {}).get(site, {})
    if site_tag_op:
        tag_lines = []
        for tag, counts in sorted(site_tag_op.items()):
            tag_lines.append(f"{tag or 'unknown'} -> {_format_counter_dist(counts)}")
        lines.append("This-site tag/op: " + "; ".join(tag_lines[:8]))

    site_labels = priors.get('site_labels', {}).get(site, [])
    if site_labels:
        formatted = [
            f"{x['op']} {x['tag']} \"{x['label']}\" ({x['count']})"
            for x in site_labels[:6]
        ]
        lines.append("This-site common labels: " + "; ".join(formatted))

    return "\n".join(lines)

def _legacy_extract_value_from_task(task: str, op: str, attrs: str) -> str:
    if op == 'CLICK':
        return ""

    task_str = str(task)

    # 1. SELECT: options matching first
    if op == 'SELECT' and 'options=' in attrs:
        options = _parse_options(attrs)
        task_lower = task_str.lower()
        # Exact substring match
        for opt in options:
            if opt.lower() in task_lower:
                return opt
        # Fuzzy: token overlap
        best_opt, best_score = None, 0
        task_words = set(re.findall(r'\w+', task_lower))
        for opt in options:
            opt_words = set(re.findall(r'\w+', opt.lower()))
            score = len(opt_words & task_words) / max(len(opt_words), 1)
            if score > best_score:
                best_score, best_opt = score, opt
        if best_opt and best_score >= 0.5:
            return best_opt

    # 2. Quoted value (SELECT도 포함 — options 매칭 실패 시 fallback)
    m = re.search(r'["\'](.*?)["\']', task_str)
    if m:
        return m.group(1)

    # 3. Field-label based extraction for TYPE
    # Parse label or placeholder from attrs, find it in task, extract following text
    if op == 'TYPE':
        attrs_dict = parse_attrs_str(attrs)
        field_label = attrs_dict.get('label', '') or attrs_dict.get('placeholder', '') or attrs_dict.get('aria-label', '')
        if field_label:
            idx = task_str.lower().find(field_label.lower())
            if idx != -1:
                after = task_str[idx + len(field_label):].strip()
                # Extract first meaningful chunk (stop at comma, period, or end)
                chunk = re.split(r'[,\.;]', after)[0].strip()
                if chunk:
                    return chunk

    # 4. Date pattern fallback
    date_m = re.search(r'\d{4}-\d{2}-\d{2}', task_str)
    if date_m:
        return date_m.group(0)

    return ""

def extract_value_from_task(task: str, op: str, attrs: str) -> str:
    op = str(op or 'CLICK').upper()
    if op == 'CLICK':
        return ""

    task_str = "" if task is None or pd.isna(task) else str(task)
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

        quoted = re.search(r'["\']([^"\']+)["\']', task_str)
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
                match = re.search(label_pattern + r'\s*(?:is|as|to|:|-)?\s*([^,.;\n]+)', task_str, flags=re.IGNORECASE)
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

def _candidate_label(c):
    label = str(c.get('text', '')).strip().lower()
    if not label:
        attrs_dict = parse_attrs_str(str(c.get('attrs', '')))
        label = str(attrs_dict.get('aria-label', '') or attrs_dict.get('placeholder', '') or attrs_dict.get('label', '')).strip().lower()
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
    task_words = _token_set(task)
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

def reset_consistency_debug():
    CONSISTENCY_DEBUG.clear()

def get_consistency_debug():
    return dict(CONSISTENCY_DEBUG)

def enforce_consistency(pred, candidates):
    """Final guardrail for op/tag/value consistency.

    Optional keys `_task` and `_row` may be supplied by callers for better repair
    scoring; private keys are removed before returning.
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

    cand_map = {str(c.get('candidate_id', '')): c for c in candidates}
    target_id = str(pred.get('target_id', ''))
    value = "" if pd.isna(pred.get('value', '')) else str(pred.get('value', ''))

    if target_id not in cand_map:
        fb = fallback_rule_based({'task': task}, candidates)
        op, target_id, value = fb['op'], str(fb['target_id']), str(fb.get('value', ''))
        CONSISTENCY_DEBUG['invalid_target_repaired'] += 1

    chosen = cand_map.get(str(target_id))
    tag = str(chosen.get('tag', '')).lower() if chosen else ''

    if op == 'TYPE' and tag not in {'input', 'textarea'}:
        replacement = _best_candidate([c for c in candidates if str(c.get('tag', '')).lower() in {'input', 'textarea'}], task)
        if replacement:
            target_id = str(replacement.get('candidate_id', ''))
            chosen = replacement
            tag = str(chosen.get('tag', '')).lower()
            value = extract_value_from_task(task, 'TYPE', str(chosen.get('attrs', ''))) or value
            CONSISTENCY_DEBUG['type_target_switched'] += 1
        else:
            op, value = 'CLICK', ''
            CONSISTENCY_DEBUG['type_downgraded_click'] += 1

    if op == 'SELECT' and tag != 'select':
        replacement = _best_candidate([c for c in candidates if str(c.get('tag', '')).lower() == 'select'], task)
        if replacement:
            target_id = str(replacement.get('candidate_id', ''))
            chosen = replacement
            tag = str(chosen.get('tag', '')).lower()
            value = extract_value_from_task(task, 'SELECT', str(chosen.get('attrs', ''))) or value
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
            replacement = _best_candidate([c for c in candidates if str(c.get('tag', '')).lower() in {'button', 'a'}], task)
            if replacement:
                target_id = str(replacement.get('candidate_id', ''))
                chosen = replacement
                tag = str(chosen.get('tag', '')).lower()
                CONSISTENCY_DEBUG['click_select_target_switched'] += 1

    if op == 'SELECT':
        options = _parse_options(str(chosen.get('attrs', ''))) if chosen else []
        fixed_value = _best_option_for_task(options, task, value)
        if fixed_value != value:
            CONSISTENCY_DEBUG['select_value_repaired'] += 1
        value = fixed_value

    if op == 'CLICK':
        if value:
            CONSISTENCY_DEBUG['click_value_cleared'] += 1
        value = ""

    return {'op': op, 'target_id': str(target_id), 'value': value}

def _removed_template_prediction(row, _unused_templates, candidates):
    raise RuntimeError("Template prediction was removed; use LLM-first inference with empirical priors.")

    """
    # Try each template variant (top-3) in frequency order
    for t in template_list:
        for c in candidates:
            c_label = _candidate_label(c)
            c_tag = c.get('tag', '')
            c_attrs = str(c.get('attrs', ''))

            # Exact match
            if c_label == t['label'] and c_tag == t['tag']:
                # [Phase 1] Cross-Validation: 기계적 매칭 방지 (단, 텍스트가 아예 없는 요소는 예외 허용)
                if t['label'] and not is_task_related(t['label']) and not is_task_related(str(c.get('text', '')).lower()):
                    continue 
                
                val = extract_value_from_task(row['task'], t['op'], c_attrs)
                conf = 1.0 if t == template_list[0] else 0.85
                return {'op': t['op'], 'target_id': c.get('candidate_id'), 'value': val}, conf

    # Fuzzy fallback: try best label match against top template
    t = template_list[0]
    t_words = t['label'].split()
    # [Phase 1] 라벨이 2단어 이하이면 Fuzzy Fallback 금지 (오탐 방지)
    if len(t_words) <= 2:
        return None, 0.0

    best_c, best_score = None, 0
    for c in candidates:
        c_label = _candidate_label(c)
        if not c_label or not t['label']:
            continue
        c_words = c_label.split()
        score = len(set(t_words) & set(c_words)) / max(len(t_words), 1)
        if score > best_score and score >= 0.5:
            if is_task_related(c_label):
                best_score, best_c = score, c

    if best_c:
        c_attrs = str(best_c.get('attrs', ''))
        val = extract_value_from_task(row['task'], t['op'], c_attrs)
        return {'op': t['op'], 'target_id': best_c.get('candidate_id'), 'value': val}, 0.75

    return None, 0.0
    """

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
    tag = str(candidate.get('tag', '')).lower()
    attrs = str(candidate.get('attrs', ''))
    score = _candidate_match_score(candidate, task)

    if op == "TYPE":
        score += 2.0 if tag in {'input', 'textarea'} else -1.0
    elif op == "SELECT":
        score += 2.0 if tag == 'select' else -1.0
        options = _parse_options(attrs)
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
    task = str(row['task'])
    op = _infer_op_from_task(task)
    target_id = candidates[0].get('candidate_id', '') if candidates else ""
    value = ""

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
        value = extract_value_from_task(task, op, str(chosen.get('attrs', '')))

    if op == "CLICK":
        value = ""

    return {'op': op, 'target_id': target_id, 'value': value}

def format_candidates_with_attrs(candidates):
    """프롬프트에 attrs 정보를 포함하여 포맷팅 (정확히 15개 모두 사용)"""
    cands_text = []
    for c in candidates:
        cands_text.append(f"- ID: {c.get('candidate_id','')}, Tag: {c.get('tag','')}, Text: {c.get('text','')}, Attrs: {c.get('attrs','')}")
    return "\n".join(cands_text)
