# -*- coding: utf-8 -*-
"""
train.csv 데이터 분석 스크립트
실행: python scripts/analyze_data.py
"""

import os
import io
import sys
import json
import re
import pandas as pd
from collections import Counter

# Windows 터미널 UTF-8 강제 출력
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data", "test.csv")

# ─────────────────────────────────────────────
# 색상 출력 헬퍼
# ─────────────────────────────────────────────
def h(text):   print(f"\n{'='*60}\n  {text}\n{'='*60}")
def sub(text): print(f"\n[{text}]")
def sep():     print("-" * 50)

# ─────────────────────────────────────────────
# HTML 타입 판별: Workflow형 vs 실제웹형
# ─────────────────────────────────────────────
def is_workflow_html(html_str):
    """
    팀원 분석: site_token으로 HTML 타입 구분 가능.
    workflow-context 혹은 completed-fields 클래스가 있으면 Workflow형.
    """
    if not isinstance(html_str, str):
        return False
    return ("workflow-context" in html_str) or ("completed-fields" in html_str)

def detect_html_type(row):
    return "workflow" if is_workflow_html(str(row.get("cleaned_html", ""))) else "real_web"

# ─────────────────────────────────────────────
# candidates에서 tag, text, attrs 파싱
# ─────────────────────────────────────────────
def parse_candidates(cand_str):
    try:
        return json.loads(cand_str)
    except Exception:
        return []

def candidate_text_empty(c):
    text = str(c.get("text", "")).strip()
    attrs = str(c.get("attrs", "")).strip()
    return (not text) and (not attrs)

# ─────────────────────────────────────────────
# value가 task에 포함되어 있는지 확인
# ─────────────────────────────────────────────
def value_in_task(task, value):
    if not value or not task:
        return False
    return str(value).lower().strip() in str(task).lower()

# ─────────────────────────────────────────────
# 메인 분석
# ─────────────────────────────────────────────
def main():
    print("\n🔍 train.csv 데이터 분석 시작...")
    df = pd.read_csv(TRAIN_PATH)

    h("0. 기본 통계")
    print(f"전체 행 수       : {len(df):,}")
    print(f"고유 site_token  : {df['site_token'].nunique()}")
    print(f"고유 task 문장   : {df['task'].nunique()}")
    print(f"고유 task 비율   : {df['task'].nunique() / len(df):.1%}")

    # ─── OP 분포 ───
    h("1. OP 분포 (전체)")
    op_counts = df["op"].value_counts()
    for op, cnt in op_counts.items():
        print(f"  {op:<8}: {cnt:>6,}개  ({cnt/len(df):.1%})")

    # ─── HTML 타입 분류 ───
    h("2. HTML 타입 분류")
    df["html_type"] = df.apply(detect_html_type, axis=1)
    html_type_counts = df["html_type"].value_counts()
    for t, cnt in html_type_counts.items():
        print(f"  {t:<12}: {cnt:>6,}개  ({cnt/len(df):.1%})")

    sub("HTML 타입별 OP 분포")
    for html_type, group in df.groupby("html_type"):
        print(f"\n  [{html_type}]")
        for op, cnt in group["op"].value_counts().items():
            print(f"    {op}: {cnt:,} ({cnt/len(group):.1%})")

    # ─── site_token별 특이사항 ───
    h("3. site_token별 분석")
    site_op = df.groupby("site_token")["op"].value_counts().unstack(fill_value=0)
    site_op["total"] = site_op.sum(axis=1)
    for col in ["CLICK", "TYPE", "SELECT"]:
        if col not in site_op.columns:
            site_op[col] = 0
    site_op["click_ratio"] = site_op["CLICK"] / site_op["total"]

    sub("CLICK 비율이 95% 이상인 site_token (학습 왜곡 주의)")
    biased = site_op[site_op["click_ratio"] >= 0.95].sort_values("click_ratio", ascending=False)
    if len(biased) > 0:
        print(biased[["CLICK", "TYPE", "SELECT", "total", "click_ratio"]].to_string())
    else:
        print("  없음")

    # ─── task 중복 분석 ───
    h("4. task 중복 분석")
    task_counts = df["task"].value_counts()
    print(f"  가장 많이 반복된 task TOP 5:")
    for task, cnt in task_counts.head(5).items():
        print(f"    [{cnt}회] {task[:80]}...")

    sub("같은 task에서 몇 개의 다른 target_id가 나오나?")
    task_target_variety = df.groupby("task")["target_id"].nunique()
    print(f"  task 1개당 평균 target_id 수: {task_target_variety.mean():.2f}")
    print(f"  가장 많은 target_id를 가진 task: {task_target_variety.max()}")
    # → 같은 task라도 history가 달라서 target이 달라짐을 보여줌

    # ─── value 분석 ───
    h("5. value 분석 (TYPE/SELECT)")
    df_type_select = df[df["op"].isin(["TYPE", "SELECT"])].copy()
    print(f"  TYPE+SELECT 전체: {len(df_type_select):,}개")

    # value가 task에 포함되어 있나?
    in_task = df_type_select.apply(
        lambda r: value_in_task(r.get("task", ""), r.get("value", "")), axis=1
    )
    print(f"  value가 task에 그대로 포함: {in_task.sum():,}개 ({in_task.mean():.1%})")
    print(f"  value가 task에 없음        : {(~in_task).sum():,}개 ({(~in_task).mean():.1%})")

    sub("value가 task에 없는 예시 5개 (정규화 필요)")
    hard = df_type_select[~in_task][["task", "value"]].head(5)
    for _, r in hard.iterrows():
        print(f"    task : {str(r['task'])[:80]}")
        print(f"    value: {r['value']}")
        sep()

    # ─── candidate_elements 분석 ───
    h("6. candidate_elements 분석")
    empty_text_counts = []
    target_positions  = []
    for _, row in df.iterrows():
        cands = parse_candidates(row.get("candidate_elements", "[]"))
        if not cands:
            continue
        # text/attrs 비어있는 후보 비율
        empty_cnt = sum(1 for c in cands if candidate_text_empty(c))
        empty_text_counts.append(empty_cnt / max(len(cands), 1))
        # 정답 후보가 몇 번째에 있나?
        target_id = str(row.get("target_id", ""))
        for i, c in enumerate(cands):
            if str(c.get("candidate_id", "")) == target_id:
                target_positions.append(i)
                break

    avg_empty = sum(empty_text_counts) / max(len(empty_text_counts), 1)
    print(f"  후보당 text/attrs 비어있는 비율 (평균): {avg_empty:.1%}")

    if target_positions:
        from collections import Counter
        pos_counter = Counter(target_positions)
        print(f"\n  정답 후보의 위치 분포 (0=첫 번째, 14=마지막):")
        for pos in sorted(pos_counter.keys()):
            bar = "█" * (pos_counter[pos] // 50)
            print(f"    위치 {pos:>2}: {pos_counter[pos]:>5}개  {bar}")

    # ─── history 분석 ───
    h("7. history 분석")
    df["history_steps"] = df["history"].apply(
        lambda h: len(re.findall(r"Step \d+:", str(h))) if isinstance(h, str) else 0
    )
    print(f"  history 평균 스텝 수: {df['history_steps'].mean():.2f}")
    print(f"  history 없음 (Step 0): {(df['history_steps'] == 0).sum():,}개")
    print(f"  history 1~3 스텝    : {((df['history_steps'] >= 1) & (df['history_steps'] <= 3)).sum():,}개")
    print(f"  history 4+ 스텝     : {(df['history_steps'] >= 4).sum():,}개")

    sub("history 스텝 수별 op 분포 (history가 많을수록 CLICK 많아지나?)")
    for step_range, label in [((0,0), "0 스텝"), ((1,3), "1~3 스텝"), ((4,99), "4+ 스텝")]:
        mask = (df["history_steps"] >= step_range[0]) & (df["history_steps"] <= step_range[1])
        group = df[mask]
        if len(group) == 0: continue
        op_dist = " | ".join(f"{op}:{cnt/len(group):.0%}" for op, cnt in group["op"].value_counts().items())
        print(f"  [{label}] {len(group):>5}행: {op_dist}")

    # ─── train vs test site_token 비교 ───
    h("8. train vs test site_token 비교")
    try:
        test_df = pd.read_csv(TEST_PATH)
        train_sites = set(df["site_token"].unique())
        test_sites  = set(test_df["site_token"].unique())
        only_train = train_sites - test_sites
        only_test  = test_sites - train_sites
        both       = train_sites & test_sites
        print(f"  train에만 있는 site_token: {len(only_train)}개  {list(only_train)[:5]}")
        print(f"  test에만 있는 site_token : {len(only_test)}개   {list(only_test)[:5]}")
        print(f"  공통 site_token          : {len(both)}개")
    except FileNotFoundError:
        print("  test.csv 없음 — 비교 생략")

    h("✅ 분석 완료!")
    print("위 결과를 바탕으로 학습 전략을 설계하세요.\n")


if __name__ == "__main__":
    main()
