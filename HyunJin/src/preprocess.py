import pandas as pd
import json
import re
from collections import Counter

def parse_attrs_str(attrs_str: str) -> dict:
    result = {}
    for part in str(attrs_str).split(' | '):
        if '=' in part:
            k, v = part.split('=', 1)
            result[k.strip()] = v.strip()
    return result

def get_history_signature(history_str):
    """History의 op 시퀀스를 시그니처로 사용하여 워크플로우 분기(Branch) 구별"""
    if pd.isna(history_str) or not str(history_str).strip():
        return "START"
    ops = re.findall(r'->\s*(CLICK|TYPE|SELECT)', str(history_str))
    if not ops:
        return "STEP_" + str(len(re.findall(r'Step \d+:', str(history_str))) + 1)
    return ",".join(ops)

def build_site_templates(train_df):
    raw = {}
    for _, row in train_df.iterrows():
        site = str(row['site_token'])
        signature = get_history_signature(row['history'])
        key = (site, signature)

        target_id = str(row['target_id'])
        cands = []
        try: cands = json.loads(row['candidate_elements'])
        except: pass
        target_cand = next((c for c in cands if str(c.get('candidate_id')) == target_id), None)

        label = target_cand.get('text', '') if target_cand else ''
        if not label and target_cand and 'attrs' in target_cand:
            attrs_dict = parse_attrs_str(target_cand['attrs'])
            label = attrs_dict.get('aria-label', '') or attrs_dict.get('placeholder', '')

        tag = target_cand.get('tag', '') if target_cand else ''
        op = row['op']

        if key not in raw:
            raw[key] = []
        raw[key].append((op, str(label).strip().lower(), tag))

    # Keep top-3 most common (op, label, tag) per key for broader coverage
    templates = {}
    for key, entries in raw.items():
        top = Counter(entries).most_common(3)
        templates[key] = [{'op': e[0][0], 'label': e[0][1], 'tag': e[0][2], 'count': e[1]} for e in top]
    return templates

def extract_value_from_task(task: str, op: str, attrs: str) -> str:
    if op == 'CLICK':
        return ""

    task_str = str(task)

    # 1. SELECT: options matching first
    if op == 'SELECT' and 'options=' in attrs:
        opts_str = attrs.split('options=')[1].split(' | ')[0]
        options = [o.strip() for o in opts_str.split(' / ') if o.strip()]
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

def _candidate_label(c):
    label = str(c.get('text', '')).strip().lower()
    if not label:
        attrs_dict = parse_attrs_str(str(c.get('attrs', '')))
        label = str(attrs_dict.get('aria-label', '') or attrs_dict.get('placeholder', '') or attrs_dict.get('label', '')).strip().lower()
    return label

def predict_by_template(row, templates, candidates):
    site = str(row['site_token'])
    signature = get_history_signature(row['history'])
    key = (site, signature)

    if key not in templates:
        return None, 0.0

    template_list = templates[key]
    task_lower = str(row['task']).lower()
    task_words = set(re.findall(r'\w+', task_lower))

    def is_task_related(label):
        if not label: return False
        l_words = set(re.findall(r'\w+', label))
        if not l_words: return False
        meaningful_l_words = {w for w in l_words if len(w) > 2}
        if not meaningful_l_words:
            meaningful_l_words = l_words
        return bool(meaningful_l_words & task_words)

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

def fallback_rule_based(row, candidates):
    task = str(row['task'])
    task_lower = task.lower()
    op = "CLICK"
    target_id = candidates[0].get('candidate_id', '') if candidates else ""
    value = ""

    def best_candidate(pool):
        """Pick candidate whose text/label best matches the task."""
        if not pool:
            return ""
        best_id, best_score = pool[0].get('candidate_id', ''), -1
        for c in pool:
            c_text = str(c.get('text', '')).lower()
            attrs_dict = parse_attrs_str(str(c.get('attrs', '')))
            c_label = (attrs_dict.get('label', '') or attrs_dict.get('placeholder', '') or attrs_dict.get('aria-label', '')).lower()
            score = 0
            for word in re.findall(r'\w+', c_text + ' ' + c_label):
                if word in task_lower and len(word) > 2:
                    score += 1
            if score > best_score:
                best_score, best_id = score, c.get('candidate_id', '')
        return best_id

    if re.search(r'\btype\b|\benter\b|\binput\b', task_lower):
        op = "TYPE"
        inputs = [c for c in candidates if c.get('tag') in ['input', 'textarea']]
        if inputs:
            target_id = best_candidate(inputs)
            matched_cand = next((c for c in inputs if c.get('candidate_id') == target_id), None)
            attrs_str = str(matched_cand.get('attrs', '')) if matched_cand else ''
            value = extract_value_from_task(task, 'TYPE', attrs_str)
    elif re.search(r'\bselect\b|\bchoose\b', task_lower):
        op = "SELECT"
        selects = [c for c in candidates if c.get('tag') == 'select']
        if selects:
            target_id = best_candidate(selects)
            matched_cand = next((c for c in selects if c.get('candidate_id') == target_id), None)
            attrs_str = str(matched_cand.get('attrs', '')) if matched_cand else ''
            value = extract_value_from_task(task, 'SELECT', attrs_str)
    else:
        m = re.search(r'click\s+(?:on\s+)?["\'](.*?)["\']', task_lower)
        if m:
            label = m.group(1)
            for c in candidates:
                if label in str(c.get('text', '')).lower():
                    target_id = c.get('candidate_id')
                    break

    return {'op': op, 'target_id': target_id, 'value': value}

def format_candidates_with_attrs(candidates):
    """프롬프트에 attrs 정보를 포함하여 포맷팅 (정확히 15개 모두 사용)"""
    cands_text = []
    for c in candidates:
        cands_text.append(f"- ID: {c.get('candidate_id','')}, Tag: {c.get('tag','')}, Text: {c.get('text','')}, Attrs: {c.get('attrs','')}")
    return "\n".join(cands_text)
