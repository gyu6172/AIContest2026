import json
import re
from collections import defaultdict

import pandas as pd

from preprocess import get_history_signature, parse_attrs_str, detect_html_type


def _tokens(text):
    return set(re.findall(r'\w+', str(text).lower()))


def _jaccard(a, b):
    a_tokens, b_tokens = _tokens(a), _tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _candidate_label(candidate):
    if not candidate:
        return ""
    text = str(candidate.get('text', '')).strip()
    if text:
        return text
    attrs = parse_attrs_str(str(candidate.get('attrs', '')))
    return str(attrs.get('label', '') or attrs.get('aria-label', '') or attrs.get('placeholder', '') or attrs.get('name', '')).strip()


class ExampleRetriever:
    def __init__(self):
        self.examples = []
        # html_type-aware 인덱스 (real_web / workflow 오염 방지)
        self.by_site_sig  = defaultdict(list)   # (site, sig, html_type) → positions
        self.by_site      = defaultdict(list)   # (site, html_type) → positions
        self.by_sig       = defaultdict(list)   # (sig, html_type) → positions
        # html_type 무관 fallback 인덱스
        self.by_site_sig_any = defaultdict(list)
        self.by_site_any     = defaultdict(list)

    def build(self, train_df):
        self.examples        = []
        self.by_site_sig     = defaultdict(list)
        self.by_site         = defaultdict(list)
        self.by_sig          = defaultdict(list)
        self.by_site_sig_any = defaultdict(list)
        self.by_site_any     = defaultdict(list)

        for idx, row in train_df.iterrows():
            try:
                candidates = json.loads(row['candidate_elements'])
            except Exception:
                candidates = []

            target_id = str(row.get('target_id', ''))
            target = next((c for c in candidates if str(c.get('candidate_id', '')) == target_id), None)
            label = _candidate_label(target)
            value = "" if pd.isna(row.get('value', '')) else str(row.get('value', ''))
            if row.get('op') == 'CLICK':
                value = ""

            html_type = detect_html_type(row)
            example = {
                'id': str(row.get('id', idx)),
                'site_token': str(row.get('site_token', '')),
                'history_signature': get_history_signature(row.get('history', '')),
                'html_type': html_type,
                'task': str(row.get('task', '')),
                'history': "" if pd.isna(row.get('history', '')) else str(row.get('history', '')),
                'target_label': label,
                'target_op': str(row.get('op', 'CLICK')),
                'target_value': value,
            }
            pos = len(self.examples)
            self.examples.append(example)

            site = example['site_token']
            sig  = example['history_signature']

            # html_type-aware 인덱스
            self.by_site_sig[(site, sig, html_type)].append(pos)
            self.by_site[(site, html_type)].append(pos)
            self.by_sig[(sig, html_type)].append(pos)
            # html_type 무관 fallback
            self.by_site_sig_any[(site, sig)].append(pos)
            self.by_site_any[site].append(pos)
        return self

    def _rank(self, indexes, row, k, exclude_id=None):
        task = str(row.get('task', ''))
        scored = []
        for idx in indexes:
            ex = self.examples[idx]
            if exclude_id is not None and str(ex.get('id')) == str(exclude_id):
                continue
            scored.append((_jaccard(task, ex['task']), idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.examples[idx] for _, idx in scored[:k]]

    def query(self, row, k=3, exclude_id=None):
        site      = str(row.get('site_token', ''))
        sig       = get_history_signature(row.get('history', ''))
        html_type = detect_html_type(row)

        # 1순위: 같은 사이트 + 같은 히스토리 패턴 + 같은 html_type
        results = self._rank(self.by_site_sig.get((site, sig, html_type), []), row, k, exclude_id)
        seen = {ex['id'] for ex in results}

        # 2순위: 같은 사이트 + 같은 html_type
        if len(results) < k:
            more = self._rank(self.by_site.get((site, html_type), []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        # 3순위: 같은 히스토리 패턴 + 같은 html_type
        if len(results) < k:
            more = self._rank(self.by_sig.get((sig, html_type), []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        # 4순위 (fallback): html_type 무관, 같은 사이트+히스토리 패턴
        if len(results) < k:
            more = self._rank(self.by_site_sig_any.get((site, sig), []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        # 5순위 (최종 fallback): html_type 무관, 같은 사이트
        if len(results) < k:
            more = self._rank(self.by_site_any.get(site, []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        return results[:k]


