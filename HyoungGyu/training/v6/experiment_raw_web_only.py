"""raw_web 전용 ranker 실험: 같은 split/feature로 raw_web만 훈련 → raw_web 검증셋 평가.
mixed 모델(48.9%)과 비교해 노이즈 가설 검증.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_lgbm_ranker_v6 import (  # noqa: E402
    RANKER_CONFIGS,
    apply_tag_mask,
    build_category_maps,
    build_rank_features,
    compute_priors,
    enforce_non_click_values,
    evaluate_predictions,
    fit_ranker,
    fit_tag_model,
    group_sizes,
    load_train_data,
    make_id_split,
    predict_from_scores,
    subset_candidates,
    subset_rows,
)


def filter_raw_web(rows: pd.DataFrame, cands: pd.DataFrame):
    rw_rows = rows[rows["html_type"] == "raw_web"].copy().reset_index(drop=True)
    rw_ids = set(rw_rows["id"].astype(str))
    rw_cands = cands[cands["id"].astype(str).isin(rw_ids)].copy().reset_index(drop=True)
    return rw_rows, rw_cands


def main() -> None:
    row_df, cand_df = load_train_data()
    train_ids, val_ids = make_id_split(row_df)

    trn_rows = subset_rows(row_df, train_ids)
    val_rows = subset_rows(row_df, val_ids)
    trn_cands = subset_candidates(cand_df, train_ids)
    val_cands = subset_candidates(cand_df, val_ids)

    trn_rows_rw, trn_cands_rw = filter_raw_web(trn_rows, trn_cands)
    val_rows_rw, val_cands_rw = filter_raw_web(val_rows, val_cands)

    print("[mixed split]")
    print("  train rows:", len(trn_rows), " val rows:", len(val_rows))
    print("[raw_web only]")
    print("  train rows:", len(trn_rows_rw), " val rows:", len(val_rows_rw))
    print("  train cands:", len(trn_cands_rw), " val cands:", len(val_cands_rw))

    print("\n[fit] tag model on raw_web only")
    tag_model, tag_encoder, tag_bundle, trn_tag_proba, val_tag_proba, tag_metric = (
        fit_tag_model(trn_rows_rw, val_rows_rw, trn_cands_rw)
    )
    print("  tag metrics:", tag_metric)

    print("\n[features] ranker on raw_web only")
    rank_category_maps = build_category_maps(trn_rows_rw, trn_cands_rw)
    site_freq = trn_rows_rw["site_key"].value_counts(normalize=True).to_dict()
    priors = compute_priors(trn_rows_rw)
    X_trn = build_rank_features(
        trn_cands_rw, trn_rows_rw, tag_encoder, trn_tag_proba,
        rank_category_maps, site_freq, priors,
    )
    X_val = build_rank_features(
        val_cands_rw, val_rows_rw, tag_encoder, val_tag_proba,
        rank_category_maps, site_freq, priors,
    )
    y_trn = trn_cands_rw["label_is_target"].astype(int).to_numpy()
    y_val = val_cands_rw["label_is_target"].astype(int).to_numpy()
    g_trn = group_sizes(trn_cands_rw)
    g_val = group_sizes(val_cands_rw)

    print("\n[fit] ranker ensemble on raw_web only")
    rankers = []
    val_score_parts = []
    for cfg in RANKER_CONFIGS:
        print(" -", cfg["name"])
        ranker = fit_ranker(X_trn, y_trn, g_trn, X_val, y_val, g_val, config=cfg, n_estimators=1800)
        scores = ranker.predict(X_val, num_iteration=ranker.best_iteration_)
        rankers.append(ranker)
        val_score_parts.append(scores)
    val_scores = np.mean(val_score_parts, axis=0)
    pred_val = predict_from_scores(val_cands_rw, val_rows_rw, val_scores, g_val)
    pred_val = enforce_non_click_values(pred_val, val_rows_rw, val_cands_rw)
    metrics = evaluate_predictions(pred_val)
    print("\n[raw_web-only model on raw_web val]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n[비교] mixed 모델의 raw_web val 성능 (이전 기록):")
    print("  raw_web_target_id_acc: 0.4889")
    print("  raw_web_all_match_acc: 0.3337")

    delta_target = metrics.get("target_id_acc", 0) - 0.4889
    delta_all = metrics.get("all_match_acc", 0) - 0.3337
    print(f"\n[Δ] raw_web 전용 - mixed:")
    print(f"  target_id_acc: {delta_target:+.4f}")
    print(f"  all_match_acc: {delta_all:+.4f}")


if __name__ == "__main__":
    main()
