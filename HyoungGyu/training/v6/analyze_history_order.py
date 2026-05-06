"""history step이 task 구문 순서대로 진행되는지 검증.
+ "다음 step(=정답)이 다음 미완료 구문에 해당하는지" 확인.
"""
import re
import json
import pandas as pd
from collections import Counter

DATA = "D:/Dev_Projects/디지털 경진대회 2026/contest_prac/data/train.csv"
df = pd.read_csv(DATA, encoding="utf-8", encoding_errors="replace")
df = df[df["history"].astype(str).str.strip().ne("") & df["history"].astype(str).ne("nan")].copy()
print(f"history 존재 row: {len(df)}")

SPLIT_RE = re.compile(r",|;|\.\s|\band\b|\bthen\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STEP_RE = re.compile(
    r"Step\s*(\d+)\s*:\s*\[([^\]]+)\]\s*(.*?)\s*->\s*([A-Za-z]+)(?::\s*(.*?))?(?=\n\s*Step|\Z)",
    re.I | re.S,
)


def clauses(t):
    t = re.sub(r"^\s*Task\s*:\s*", "", str(t), flags=re.I).strip()
    return [c.strip() for c in SPLIT_RE.split(t) if c.strip()]


def toks(s):
    return set(t.lower() for t in TOKEN_RE.findall(str(s)))


def parse_history(h):
    out = []
    for m in STEP_RE.finditer(str(h)):
        text = (m.group(3) or "")
        val = (m.group(5) or "")
        op = m.group(4) or ""
        tag = m.group(2) or ""
        out.append({"text": text, "val": val, "op": op.upper(), "tag": tag})
    return out


def best_clause_idx(text, val, clause_list):
    s = toks(text) | toks(val)
    if not s:
        return -1, 0
    best_i, best_score = -1, 0
    for i, c in enumerate(clause_list):
        score = len(s & toks(c))
        if score > best_score:
            best_score, best_i = score, i
    return best_i, best_score


multi_step_rows = 0
positional_strict = 0
monotonic = 0
target_aligned = 0
target_clause_skipped = 0
n_clauses_dist = Counter()
step_count_dist = Counter()

# 더 본 케이스: row마다 step별 어느 절을 매칭했는지 + 정답이 다음 절인지
sample_aligned = []
sample_misaligned = []

for _, row in df.iterrows():
    cls = clauses(row["task"])
    hist = parse_history(row["history"])
    if len(cls) < 2 or len(hist) < 2:
        continue
    multi_step_rows += 1
    n_clauses_dist[len(cls)] += 1
    step_count_dist[len(hist)] += 1

    matched_idx = []
    for step in hist:
        i, sc = best_clause_idx(step["text"], step["val"], cls)
        matched_idx.append(i if sc >= 1 else -1)

    valid_matches = [i for i in matched_idx if i >= 0]

    # 1) Strict positional: step_i가 clause_i와 매칭되는가
    strict_ok = sum(1 for k, i in enumerate(matched_idx) if i == k)
    if strict_ok == len(hist):
        positional_strict += 1

    # 2) Monotonic: 매칭된 인덱스가 단조증가
    if len(valid_matches) >= 2:
        if all(valid_matches[k] < valid_matches[k + 1] for k in range(len(valid_matches) - 1)):
            monotonic += 1
    elif len(valid_matches) == 1:
        monotonic += 1
    else:
        pass  # no match — skip

    # 3) 정답이 "다음 미완료 절"과 매칭되는가
    if valid_matches:
        next_clause_idx = max(valid_matches) + 1
        if next_clause_idx >= len(cls):
            target_clause_skipped += 1
            continue
        next_clause_toks = toks(cls[next_clause_idx])
        # candidate_elements에서 정답 후보의 text/attrs 토큰 모음
        try:
            cands = json.loads(row["candidate_elements"])
        except Exception:
            continue
        target_id = row.get("target_id")
        true_c = next((c for c in cands if c.get("candidate_id") == target_id), None)
        if not true_c:
            continue
        ttoks = toks(str(true_c.get("text", "")) + " " + str(true_c.get("attrs", "")))
        op = row.get("op", "")
        val = str(row.get("value", "")) if pd.notna(row.get("value")) else ""
        ttoks |= toks(val)
        overlap = len(ttoks & next_clause_toks)
        if overlap >= 1:
            target_aligned += 1
            if len(sample_aligned) < 3:
                sample_aligned.append({
                    "task": row["task"][:120],
                    "matched_idx": matched_idx,
                    "next_clause": cls[next_clause_idx][:80],
                    "target_text": str(true_c.get("text", ""))[:60],
                    "value": val[:40],
                    "op": op,
                })
        else:
            if len(sample_misaligned) < 3:
                sample_misaligned.append({
                    "task": row["task"][:120],
                    "matched_idx": matched_idx,
                    "next_clause": cls[next_clause_idx][:80],
                    "target_text": str(true_c.get("text", ""))[:60],
                    "value": val[:40],
                    "op": op,
                })

denom = max(multi_step_rows, 1)
print(f"\n분석 가능한 row(2절+ AND 2step+): {multi_step_rows}")
print(f"  positional strict (step_i ↔ clause_i): {positional_strict} ({positional_strict/denom*100:.1f}%)")
print(f"  monotonic (매칭된 절 인덱스 단조증가): {monotonic} ({monotonic/denom*100:.1f}%)")
print(f"  next_clause가 task 끝을 넘어감(스킵): {target_clause_skipped}")
print(f"  '정답이 다음 절'과 토큰 매칭됨: {target_aligned} ({target_aligned/denom*100:.1f}%)")

# op별로도 보자
print("\n[op별로 '정답=다음 절' 매칭 비율]")
for op_target in ["CLICK", "TYPE", "SELECT"]:
    cnt = aligned = 0
    for _, row in df.iterrows():
        if str(row.get("op", "")) != op_target:
            continue
        cls = clauses(row["task"])
        hist = parse_history(row["history"])
        if len(cls) < 2 or len(hist) < 2:
            continue
        matched_idx = []
        for step in hist:
            i, sc = best_clause_idx(step["text"], step["val"], cls)
            matched_idx.append(i if sc >= 1 else -1)
        valid = [i for i in matched_idx if i >= 0]
        if not valid:
            continue
        next_clause_idx = max(valid) + 1
        if next_clause_idx >= len(cls):
            continue
        next_clause_toks = toks(cls[next_clause_idx])
        try:
            cands = json.loads(row["candidate_elements"])
        except Exception:
            continue
        true_c = next((c for c in cands if c.get("candidate_id") == row.get("target_id")), None)
        if not true_c:
            continue
        ttoks = toks(str(true_c.get("text", "")) + " " + str(true_c.get("attrs", "")))
        ttoks |= toks(str(row.get("value", "")) if pd.notna(row.get("value")) else "")
        cnt += 1
        if len(ttoks & next_clause_toks) >= 1:
            aligned += 1
    if cnt:
        print(f"  {op_target}: {aligned}/{cnt} = {aligned/cnt*100:.1f}%")

print("\n[정렬된 샘플]")
for s in sample_aligned:
    print(s)
print("\n[정렬 안 된 샘플]")
for s in sample_misaligned:
    print(s)
