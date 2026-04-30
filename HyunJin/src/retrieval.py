import json
import re
from collections import defaultdict

import pandas as pd

from preprocess import get_history_signature, parse_attrs_str


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
        self.by_site_sig = defaultdict(list)
        self.by_site = defaultdict(list)
        self.by_sig = defaultdict(list)

    def build(self, train_df):
        self.examples = []
        self.by_site_sig = defaultdict(list)
        self.by_site = defaultdict(list)
        self.by_sig = defaultdict(list)

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

            example = {
                'id': str(row.get('id', idx)),
                'site_token': str(row.get('site_token', '')),
                'history_signature': get_history_signature(row.get('history', '')),
                'task': str(row.get('task', '')),
                'history': "" if pd.isna(row.get('history', '')) else str(row.get('history', '')),
                'target_label': label,
                'target_op': str(row.get('op', 'CLICK')),
                'target_value': value,
            }
            pos = len(self.examples)
            self.examples.append(example)
            self.by_site_sig[(example['site_token'], example['history_signature'])].append(pos)
            self.by_site[example['site_token']].append(pos)
            self.by_sig[example['history_signature']].append(pos)
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
        site = str(row.get('site_token', ''))
        sig = get_history_signature(row.get('history', ''))

        results = self._rank(self.by_site_sig.get((site, sig), []), row, k, exclude_id)
        seen = {ex['id'] for ex in results}

        if len(results) < k:
            more = self._rank(self.by_site.get(site, []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        if len(results) < k:
            more = self._rank(self.by_sig.get(sig, []), row, k * 3, exclude_id)
            for ex in more:
                if ex['id'] not in seen:
                    results.append(ex)
                    seen.add(ex['id'])
                if len(results) >= k:
                    break

        return results[:k]


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
