import pandas as pd
import numpy as np
import json
import joblib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "HyoungGyu" / "training" / "v6"))

bundle = joblib.load(ROOT / "HyoungGyu/training/v6/artifacts/lgbm_ranker_v6_bundle.joblib")
tag_model = bundle["tag_model"]
tag_encoder = bundle["tag_encoder"]
tag_bundle = bundle["tag_bundle"]

row_df = pd.read_csv(ROOT / "HyoungGyu/preprocessing/preprocessed_train_custom.csv", low_memory=False, encoding="utf-8", encoding_errors="replace")
cand_df = pd.read_csv(ROOT / "HyoungGyu/preprocessing/preprocessed_train_candidates_custom.csv", low_memory=False, encoding="utf-8", encoding_errors="replace")
val = pd.read_csv(ROOT / "HyoungGyu/training/v6/artifacts/validation_predictions.csv", encoding="utf-8", encoding_errors="replace")

val_ids = set(val["id"].astype(str))
row_v = row_df[row_df["id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)
cand_v = cand_df[cand_df["id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)

raw_rows = row_v[row_v["html_type"] == "raw_web"].copy().reset_index(drop=True)
raw_ids = set(raw_rows["id"].astype(str))
raw_cands = cand_v[cand_v["id"].astype(str).isin(raw_ids)].copy().reset_index(drop=True)
print("raw_web rows:", len(raw_rows), "cands:", len(raw_cands))

tag_counts_per_row = raw_cands.groupby("id").apply(
    lambda g: int((g["candidate_tag"] == g["target_tag"]).sum()),
    include_groups=False,
)
print("\nraw_web 정답 tag와 같은 후보 수 분포:")
print(tag_counts_per_row.value_counts().sort_index())
unique_target_tag = int((tag_counts_per_row == 1).sum())
total_raw = len(raw_rows)
print(f"\n정답 tag가 unique한 row: {unique_target_tag}/{total_raw} = {unique_target_tag/total_raw*100:.1f}%")

wf_rows = row_v[row_v["html_type"] == "workflow"]
wf_ids = set(wf_rows["id"].astype(str))
wf_cands = cand_v[cand_v["id"].astype(str).isin(wf_ids)]
wf_tag_counts = wf_cands.groupby("id").apply(
    lambda g: int((g["candidate_tag"] == g["target_tag"]).sum()),
    include_groups=False,
)
wf_unique = int((wf_tag_counts == 1).sum())
print(f"workflow 정답 tag unique: {wf_unique}/{len(wf_rows)} = {wf_unique/len(wf_rows)*100:.1f}%")
print(f"\nraw_web 정답 tag 동일 후보 평균: {tag_counts_per_row.mean():.2f}")
print(f"workflow 평균: {wf_tag_counts.mean():.2f}")

print("\nraw_web target_tag 분포:")
print(raw_rows["target_tag"].value_counts().head(10))


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9/._:+,-]*[A-Za-z0-9])?")
DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
CODE_RE = re.compile(r"\b[A-Z]{1,6}-(?=[A-Z0-9]*\d)[A-Z0-9]{2,20}\b", re.I)
TYPE_HINTS = {"address","amount","asset","caller","card","city","code","date","email","id","manager","name","number","owner","phone","price","reviewer","serial","time","title","url"}
SELECT_HINTS = {"choose","dropdown","locale","mark","pick","priority","select","status"}
CLICK_HINTS = {"approve","cancel","click","open","publish","queue","save","send","submit"}


def jl(v, default):
    if isinstance(v, (list, dict)):
        return v
    s = str(v) if v is not None else ""
    if not s or s == "nan":
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def encode(v, m):
    return m.get(str(v) if v is not None else "", len(m))


def safe_ratio(n, d):
    return float(n / d) if d else 0.0


def build_tag_features(rdf, vocab, cat_maps, site_freq, top_tags):
    rows = []
    token_to_col = {tok: f"tok_{i:03d}" for i, tok in enumerate(vocab)}
    for r in rdf.itertuples(index=False):
        task_tokens = set(jl(getattr(r, "task_tokens_json"), []))
        task_text = str(getattr(r, "task_clean"))
        tag_counts = jl(getattr(r, "candidate_tag_counts_json"), {})
        if not isinstance(tag_counts, dict):
            tag_counts = {}
        feats = {
            "site_idx": encode(getattr(r, "site_key"), cat_maps["site_key"]),
            "site_freq": float(site_freq.get(str(getattr(r, "site_key")), 0.0)),
            "html_type_idx": encode(getattr(r, "html_type"), cat_maps["html_type"]),
            "is_workflow": int(getattr(r, "is_workflow")),
            "history_step_count": int(getattr(r, "history_step_count")),
            "history_last_op_idx": encode(getattr(r, "history_last_op"), cat_maps["history_last_op"]),
            "html_current_step": int(getattr(r, "html_current_step")),
            "html_total_steps": int(getattr(r, "html_total_steps")),
            "html_step_remaining": int(getattr(r, "html_step_remaining")),
            "html_progress": safe_ratio(int(getattr(r, "html_current_step")), int(getattr(r, "html_total_steps"))),
            "n_candidates": int(getattr(r, "n_candidates")),
            "candidate_unique_tag_count": int(getattr(r, "candidate_unique_tag_count")),
            "candidate_max_same_tag_count": int(getattr(r, "candidate_max_same_tag_count")),
            "task_token_count": len(task_tokens),
            "task_char_len": len(task_text),
            "task_number_count": len(DECIMAL_RE.findall(task_text)),
            "task_code_count": len(CODE_RE.findall(task_text)),
            "task_kw_type": len(task_tokens & TYPE_HINTS),
            "task_kw_select": len(task_tokens & SELECT_HINTS),
            "task_kw_click": len(task_tokens & CLICK_HINTS),
        }
        for tag in top_tags:
            try:
                feats[f"cand_tag_count_{tag}"] = int(tag_counts.get(tag, 0))
            except Exception:
                feats[f"cand_tag_count_{tag}"] = 0
        for tok, col in token_to_col.items():
            feats[col] = int(tok in task_tokens)
        rows.append(feats)
    return pd.DataFrame(rows).fillna(0)


X_raw = build_tag_features(
    raw_rows,
    tag_bundle["token_vocab"],
    tag_bundle["category_maps"],
    tag_bundle["site_freq"],
    tag_bundle["top_candidate_tags"],
)
feat_names = tag_bundle["feature_names"]
for c in feat_names:
    if c not in X_raw.columns:
        X_raw[c] = 0
X_raw = X_raw[feat_names]
proba = tag_model.predict_proba(X_raw)
pred_tag = tag_encoder.classes_[np.argmax(proba, axis=1)]
raw_rows["pred_tag"] = pred_tag
tag_acc_raw = float((raw_rows["pred_tag"] == raw_rows["target_tag"]).mean())
print(f"\nraw_web tag 모델 top1 정확도: {tag_acc_raw*100:.1f}%")

raw_cands_g = {rid: g for rid, g in raw_cands.groupby("id", sort=False)}

correct_simple = 0
in_set = 0
narrowed_zero = 0
for _, r in raw_rows.iterrows():
    rid = r["id"]
    pt = r["pred_tag"]
    cs = raw_cands_g.get(rid)
    if cs is None:
        continue
    matched = cs[cs["candidate_tag"] == pt]
    if len(matched) == 0:
        narrowed_zero += 1
        if len(cs) > 0 and cs.iloc[0]["candidate_id"] == r["target_id"]:
            correct_simple += 1
        continue
    if r["target_id"] in matched["candidate_id"].values:
        in_set += 1
    if matched.iloc[0]["candidate_id"] == r["target_id"]:
        correct_simple += 1

print(f"\npred_tag 매칭 후보 0개인 row: {narrowed_zero}")
print(f"[베이스라인] pred_tag 좁힘+첫째 선택: {correct_simple}/{len(raw_rows)} = {correct_simple/len(raw_rows)*100:.1f}%")
print(f"[상한] pred_tag로 좁힌 후보 안에 정답 존재: {in_set}/{len(raw_rows)} = {in_set/len(raw_rows)*100:.1f}%")

v6_raw = val[val["html_type"] == "raw_web"]
v6_acc = float((v6_raw["target_id"] == v6_raw["true_target_id"]).mean())
print(f"[현재 v6 ranker] raw_web target_id 정확도: {v6_acc*100:.1f}%")

buckets = []
for _, r in raw_rows.iterrows():
    rid = r["id"]
    pt = r["pred_tag"]
    cs = raw_cands_g.get(rid)
    if cs is None:
        continue
    matched = cs[cs["candidate_tag"] == pt]
    buckets.append(len(matched))
buckets = pd.Series(buckets)
print(f"\npred_tag와 일치하는 후보 수 분포 (raw_web):")
print(buckets.value_counts().sort_index().head(15))
print(f"평균: {buckets.mean():.2f}")
print(f"1개(즉시 결정): {(buckets==1).sum()} ({(buckets==1).mean()*100:.1f}%)")
print(f"2~5개: {((buckets>=2)&(buckets<=5)).sum()} ({((buckets>=2)&(buckets<=5)).mean()*100:.1f}%)")
print(f"6개+: {(buckets>=6).sum()} ({(buckets>=6).mean()*100:.1f}%)")

ceiling_optimal_tag = int((tag_counts_per_row == 1).sum())
print(f"\n[궁극 상한] tag 모델이 100% 정확하다 가정하고 정답 tag가 unique한 row: {ceiling_optimal_tag}/{total_raw} = {ceiling_optimal_tag/total_raw*100:.1f}%")
print("→ 그 외 row는 tag 만으로는 결정 불가 (같은 tag 후보가 여러 개)")
