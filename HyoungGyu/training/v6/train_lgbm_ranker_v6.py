# somenna	0.5917	0.7287	0.9520	0.8449	98.6%

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RNG_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (
            candidate / "HyoungGyu" / "preprocessing"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root containing data/ and HyoungGyu/preprocessing/."
    )


ROOT = find_project_root(SCRIPT_DIR)
DATA_DIR = ROOT / "data"
PREPROCESS_DIR = ROOT / "HyoungGyu" / "preprocessing"
OUTPUT_DIR = SCRIPT_DIR / "artifacts"

TRAIN_ROW_PATH = PREPROCESS_DIR / "preprocessed_train_custom.csv"
TRAIN_CAND_PATH = PREPROCESS_DIR / "preprocessed_train_candidates_custom.csv"
TEST_SOURCE_CANDIDATES = [
    DATA_DIR / "test.csv",
    PREPROCESS_DIR / "preprocessed_test_v6_sequence.csv",
    PREPROCESS_DIR / "preprocessed_test_v5.csv",
]
TEST_ROW_OUTPUT = OUTPUT_DIR / "preprocessed_test_custom_v6.csv"
TEST_CAND_OUTPUT = OUTPUT_DIR / "preprocessed_test_candidates_custom_v6.csv"

SUBMISSION_PATH = SCRIPT_DIR / "submission.csv"
WORKFLOW_ONLY_SUBMISSION_PATH = (
    SCRIPT_DIR / "submission_workflow_only_rawweb_garbage.csv"
)

TAG_TOKEN_VOCAB_SIZE = 450
TOP_TAG_FEATURES = 40
TAG_MODEL_PARAMS = {
    "objective": "multiclass",
    "n_estimators": 900,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "min_child_samples": 15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RNG_SEED,
    "verbose": -1,
}
RANKER_CONFIGS = [
    {
        "name": "ranker_seed42",
        "random_state": 42,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "name": "ranker_seed77",
        "random_state": 77,
        "learning_rate": 0.03,
        "num_leaves": 95,
        "min_child_samples": 18,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
    },
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9/._:+,-]*[A-Za-z0-9])?")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CODE_RE = re.compile(r"\b[A-Z]{1,6}-(?=[A-Z0-9]*\d)[A-Z0-9]{2,20}\b", re.I)
MONEY_RE = re.compile(r"[$€£]\s*(\d+(?:\.\d+)?)")
DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

CLICK_INPUT_TYPES = {"button", "checkbox", "image", "radio", "range", "reset", "submit"}
CLICK_TAGS = {"a", "button", "div", "img", "label", "li", "p", "span", "svg", "td"}

TYPE_HINTS = {
    "address",
    "amount",
    "asset",
    "caller",
    "card",
    "city",
    "code",
    "date",
    "email",
    "id",
    "manager",
    "name",
    "number",
    "owner",
    "phone",
    "price",
    "reviewer",
    "serial",
    "time",
    "title",
    "url",
}
SELECT_HINTS = {
    "choose",
    "dropdown",
    "locale",
    "mark",
    "pick",
    "priority",
    "select",
    "status",
}
CLICK_HINTS = {
    "approve",
    "cancel",
    "click",
    "open",
    "publish",
    "queue",
    "save",
    "send",
    "submit",
}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = as_text(value)
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def tokenize(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(as_text(value))]


def norm_key(value: Any) -> str:
    text = as_text(value).replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, encoding="utf-8", encoding_errors="replace", low_memory=False
    )


def load_train_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_ROW_PATH.exists() or not TRAIN_CAND_PATH.exists():
        raise FileNotFoundError(
            "Missing custom preprocessing outputs. Run HyoungGyu/preprocessing/preprocess_train_custom.py first."
        )
    row_df = read_csv(TRAIN_ROW_PATH)
    cand_df = read_csv(TRAIN_CAND_PATH)
    valid_row_mask = (
        row_df["is_workflow"].astype(str).isin({"0", "1", "0.0", "1.0"})
        & row_df["n_candidates"].astype(str).str.fullmatch(r"\d+(?:\.0)?", na=False)
        & row_df["target_id"].astype(str).str.startswith("elem_")
        & row_df["target_tag"].fillna("").astype(str).ne("")
    )
    dropped = int((~valid_row_mask).sum())
    if dropped:
        print(f"dropped malformed train rows: {dropped}")
    row_df = row_df.loc[valid_row_mask].copy()

    if (DATA_DIR / "test.csv").exists() and "site_token_raw" in row_df.columns:
        test_sites = set(
            pd.read_csv(
                DATA_DIR / "test.csv",
                usecols=["site_token"],
                encoding="utf-8",
                encoding_errors="replace",
            )["site_token"].astype(str)
        )
        train_only_mask = ~row_df["site_token_raw"].astype(str).isin(test_sites)
        train_only_sites = sorted(
            row_df.loc[train_only_mask, "site_token_raw"].astype(str).unique()
        )
        if train_only_sites:
            print(
                f"dropped train-only sites: {train_only_sites} rows={int(train_only_mask.sum())}"
            )
            row_df = row_df.loc[~train_only_mask].copy()

    valid_ids = set(row_df["id"].astype(str))
    cand_df = cand_df[cand_df["id"].astype(str).isin(valid_ids)].copy()

    for col in [
        "is_workflow",
        "history_step_count",
        "html_current_step",
        "html_total_steps",
        "html_step_remaining",
        "n_candidates",
        "candidate_unique_tag_count",
        "candidate_max_same_tag_count",
        "target_tag_count_in_candidates",
        "target_tag_is_unique",
    ]:
        if col in row_df.columns:
            row_df[col] = (
                pd.to_numeric(row_df[col], errors="coerce").fillna(0).astype(int)
            )
    for col in [
        "candidate_idx",
        "candidate_tag_count_in_row",
        "candidate_text_empty",
        "candidate_attrs_empty",
        "candidate_options_count",
        "label_is_target",
        "label_same_as_target_tag",
        "is_workflow",
        "history_step_count",
        "html_current_step",
        "html_total_steps",
    ]:
        if col in cand_df.columns:
            cand_df[col] = (
                pd.to_numeric(cand_df[col], errors="coerce").fillna(0).astype(int)
            )
    if "candidate_tag_frac_in_row" in cand_df.columns:
        cand_df["candidate_tag_frac_in_row"] = pd.to_numeric(
            cand_df["candidate_tag_frac_in_row"], errors="coerce"
        ).fillna(0.0)
    if "op" in row_df.columns:
        valid_ops = {"CLICK", "TYPE", "SELECT"}
        row_df.loc[~row_df["op"].fillna("").astype(str).isin(valid_ops), "op"] = ""
    return row_df, cand_df


def build_test_custom_from_source() -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    sys.path.insert(0, str(PREPROCESS_DIR))
    from preprocess_train_custom import build_records

    source_path = next((path for path in TEST_SOURCE_CANDIDATES if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "Could not find a test source. Expected data/test.csv, preprocessed_test_v6_sequence.csv, or preprocessed_test_v5.csv."
        )

    test_source = read_csv(source_path)
    base_cols = [
        "id",
        "site_token",
        "task",
        "history",
        "cleaned_html",
        "candidate_elements",
    ]
    missing = [col for col in base_cols if col not in test_source.columns]
    if missing:
        raise ValueError(f"Test source {source_path} is missing columns: {missing}")
    test_source = test_source[base_cols].copy()

    row_df, cand_df, _ = build_records(test_source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(TEST_ROW_OUTPUT, index=False, encoding="utf-8", lineterminator="\n")
    cand_df.to_csv(TEST_CAND_OUTPUT, index=False, encoding="utf-8", lineterminator="\n")
    return row_df, cand_df, source_path


def load_or_build_test_data() -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    if (DATA_DIR / "test.csv").exists():
        return build_test_custom_from_source()
    if TEST_ROW_OUTPUT.exists() and TEST_CAND_OUTPUT.exists():
        return read_csv(TEST_ROW_OUTPUT), read_csv(TEST_CAND_OUTPUT), None
    return build_test_custom_from_source()


def make_id_split(row_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stratify = None
    if "op" in row_df.columns and row_df["op"].nunique() > 1:
        counts = row_df["op"].fillna("").astype(str).value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = row_df["op"].fillna("").astype(str)
    train_ids, val_ids = train_test_split(
        row_df["id"].to_numpy(),
        test_size=0.2,
        random_state=RNG_SEED,
        stratify=stratify,
    )
    return train_ids, val_ids


def subset_rows(row_df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    order = {row_id: pos for pos, row_id in enumerate(ids)}
    out = row_df[row_df["id"].isin(order)].copy()
    out["_order"] = out["id"].map(order)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def subset_candidates(cand_df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    order = {row_id: pos for pos, row_id in enumerate(ids)}
    out = cand_df[cand_df["id"].isin(order)].copy()
    out["_order"] = out["id"].map(order)
    out = out.sort_values(["_order", "candidate_idx"]).drop(columns="_order")
    return out.reset_index(drop=True)


def build_token_vocab(
    row_df: pd.DataFrame, size: int = TAG_TOKEN_VOCAB_SIZE
) -> list[str]:
    counter: Counter[str] = Counter()
    for value in row_df["task_tokens_json"].fillna("[]"):
        counter.update(json_loads(value, []))
    return [token for token, _ in counter.most_common(size)]


def build_category_maps(
    row_df: pd.DataFrame, cand_df: pd.DataFrame
) -> dict[str, dict[str, int]]:
    sources = {
        "site_key": row_df["site_key"],
        "html_type": row_df["html_type"],
        "history_last_op": row_df["history_last_op"],
        "candidate_tag": cand_df["candidate_tag"],
        "candidate_role": cand_df["candidate_role"],
        "candidate_type": cand_df["candidate_type"],
        "candidate_predicted_op": cand_df["candidate_predicted_op"],
    }
    maps: dict[str, dict[str, int]] = {}
    for name, values in sources.items():
        unique = sorted({as_text(value) for value in values.fillna("")})
        maps[name] = {value: idx for idx, value in enumerate(unique)}
    return maps


def encode_value(value: Any, mapping: dict[str, int]) -> int:
    text = as_text(value)
    return mapping.get(text, len(mapping))


def keyword_count(tokens: set[str], hints: set[str]) -> int:
    return len(tokens & hints)


def parse_tag_counts(value: Any) -> dict[str, int]:
    raw = json_loads(value, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, val in raw.items():
        try:
            out[as_text(key)] = int(val)
        except Exception:
            out[as_text(key)] = 0
    return out


def build_tag_features(
    row_df: pd.DataFrame,
    token_vocab: list[str],
    category_maps: dict[str, dict[str, int]],
    site_freq: dict[str, float],
    top_candidate_tags: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    token_to_col = {token: f"tok_{idx:03d}" for idx, token in enumerate(token_vocab)}
    for row in row_df.itertuples(index=False):
        task_tokens = set(json_loads(getattr(row, "task_tokens_json"), []))
        task_text = as_text(getattr(row, "task_clean"))
        tag_counts = parse_tag_counts(getattr(row, "candidate_tag_counts_json"))
        features: dict[str, float | int] = {
            "site_idx": encode_value(
                getattr(row, "site_key"), category_maps["site_key"]
            ),
            "site_freq": float(site_freq.get(as_text(getattr(row, "site_key")), 0.0)),
            "html_type_idx": encode_value(
                getattr(row, "html_type"), category_maps["html_type"]
            ),
            "is_workflow": int(getattr(row, "is_workflow")),
            "history_step_count": int(getattr(row, "history_step_count")),
            "history_last_op_idx": encode_value(
                getattr(row, "history_last_op"), category_maps["history_last_op"]
            ),
            "html_current_step": int(getattr(row, "html_current_step")),
            "html_total_steps": int(getattr(row, "html_total_steps")),
            "html_step_remaining": int(getattr(row, "html_step_remaining")),
            "html_progress": safe_ratio(
                int(getattr(row, "html_current_step")),
                int(getattr(row, "html_total_steps")),
            ),
            "n_candidates": int(getattr(row, "n_candidates")),
            "candidate_unique_tag_count": int(
                getattr(row, "candidate_unique_tag_count")
            ),
            "candidate_max_same_tag_count": int(
                getattr(row, "candidate_max_same_tag_count")
            ),
            "task_token_count": len(task_tokens),
            "task_char_len": len(task_text),
            "task_number_count": len(DECIMAL_RE.findall(task_text)),
            "task_code_count": len(CODE_RE.findall(task_text)),
            "task_kw_type": keyword_count(task_tokens, TYPE_HINTS),
            "task_kw_select": keyword_count(task_tokens, SELECT_HINTS),
            "task_kw_click": keyword_count(task_tokens, CLICK_HINTS),
        }
        for tag in top_candidate_tags:
            features[f"cand_tag_count_{tag}"] = tag_counts.get(tag, 0)
        for token, col in token_to_col.items():
            features[col] = int(token in task_tokens)
        rows.append(features)
    return pd.DataFrame(rows).fillna(0)


def fit_tag_model(
    trn_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    all_cands_for_maps: pd.DataFrame,
) -> tuple[
    lgb.LGBMClassifier,
    LabelEncoder,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    category_maps = build_category_maps(trn_rows, all_cands_for_maps)
    token_vocab = build_token_vocab(trn_rows)
    site_freq = trn_rows["site_key"].value_counts(normalize=True).to_dict()
    top_candidate_tags = [
        tag
        for tag, _ in Counter(
            all_cands_for_maps["candidate_tag"].fillna("").astype(str)
        ).most_common(TOP_TAG_FEATURES)
        if tag
    ]

    label_encoder = LabelEncoder()
    y_trn = label_encoder.fit_transform(trn_rows["target_tag"].fillna("").astype(str))
    class_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    y_val = np.array(
        [
            class_to_idx.get(label, -1)
            for label in val_rows["target_tag"].fillna("").astype(str)
        ],
        dtype=int,
    )

    X_trn = build_tag_features(
        trn_rows, token_vocab, category_maps, site_freq, top_candidate_tags
    )
    X_val = build_tag_features(
        val_rows, token_vocab, category_maps, site_freq, top_candidate_tags
    )

    model = lgb.LGBMClassifier(
        num_class=len(label_encoder.classes_), **TAG_MODEL_PARAMS
    )
    model.fit(X_trn, y_trn)
    val_proba = model.predict_proba(X_val)
    trn_proba = model.predict_proba(X_trn)

    metrics = tag_metrics(y_val, val_proba)
    bundle = {
        "token_vocab": token_vocab,
        "category_maps": category_maps,
        "site_freq": site_freq,
        "top_candidate_tags": top_candidate_tags,
        "feature_names": list(X_trn.columns),
    }
    return model, label_encoder, bundle, trn_proba, val_proba, metrics


def fit_final_tag_model(
    row_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> tuple[lgb.LGBMClassifier, LabelEncoder, dict[str, Any], np.ndarray]:
    category_maps = build_category_maps(row_df, cand_df)
    token_vocab = build_token_vocab(row_df)
    site_freq = row_df["site_key"].value_counts(normalize=True).to_dict()
    top_candidate_tags = [
        tag
        for tag, _ in Counter(
            cand_df["candidate_tag"].fillna("").astype(str)
        ).most_common(TOP_TAG_FEATURES)
        if tag
    ]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(row_df["target_tag"].fillna("").astype(str))
    X = build_tag_features(
        row_df, token_vocab, category_maps, site_freq, top_candidate_tags
    )
    model = lgb.LGBMClassifier(
        num_class=len(label_encoder.classes_),
        **{**TAG_MODEL_PARAMS, "n_estimators": 500},
    )
    model.fit(X, y)
    proba = model.predict_proba(X)
    bundle = {
        "token_vocab": token_vocab,
        "category_maps": category_maps,
        "site_freq": site_freq,
        "top_candidate_tags": top_candidate_tags,
        "feature_names": list(X.columns),
    }
    return model, label_encoder, bundle, proba


def predict_tag_proba(
    model: lgb.LGBMClassifier,
    label_encoder: LabelEncoder,
    bundle: dict[str, Any],
    row_df: pd.DataFrame,
) -> np.ndarray:
    X = build_tag_features(
        row_df,
        bundle["token_vocab"],
        bundle["category_maps"],
        bundle["site_freq"],
        bundle["top_candidate_tags"],
    )
    return model.predict_proba(X, num_iteration=getattr(model, "best_iteration_", None))


def tag_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(proba, axis=1)
    order = np.argsort(-proba, axis=1)
    return {
        "tag_model_acc": float((pred == y_true).mean()),
        "tag_model_top3_acc": float(
            np.mean([y_true[i] in order[i, :3] for i in range(len(y_true))])
        ),
        "tag_model_top5_acc": float(
            np.mean([y_true[i] in order[i, :5] for i in range(len(y_true))])
        ),
    }


def compute_priors(row_df: pd.DataFrame) -> dict[str, Any]:
    global_counts = row_df["target_tag"].fillna("").astype(str).value_counts()
    n_rows = len(row_df)
    global_prior = {tag: float(count / n_rows) for tag, count in global_counts.items()}

    site_tag_counts: dict[tuple[str, str], int] = defaultdict(int)
    site_counts: Counter[str] = Counter()
    for row in row_df[["site_key", "target_tag"]].itertuples(index=False):
        site = as_text(row.site_key)
        tag = as_text(row.target_tag)
        site_counts[site] += 1
        site_tag_counts[(site, tag)] += 1

    site_prior = {
        key: float((count + 1) / (site_counts[key[0]] + max(len(global_prior), 1)))
        for key, count in site_tag_counts.items()
    }
    return {
        "global_prior": global_prior,
        "site_prior": site_prior,
        "site_counts": dict(site_counts),
    }


def make_tag_proba_by_id(
    row_df: pd.DataFrame, proba: np.ndarray
) -> dict[str, np.ndarray]:
    return {as_text(row_id): proba[idx] for idx, row_id in enumerate(row_df["id"])}


def build_row_context(
    row_df: pd.DataFrame, label_encoder: LabelEncoder, tag_proba: np.ndarray
) -> dict[str, dict[str, Any]]:
    proba_by_id = make_tag_proba_by_id(row_df, tag_proba)
    out: dict[str, dict[str, Any]] = {}
    for row in row_df.itertuples(index=False):
        row_id = as_text(getattr(row, "id"))
        task_tokens = {
            token.lower() for token in json_loads(getattr(row, "task_tokens_json"), [])
        }
        panel_names = {
            norm_key(value)
            for value in json_loads(getattr(row, "html_panel_names_json"), [])
        }
        panel_labels = {
            norm_key(value)
            for value in json_loads(getattr(row, "html_panel_labels_json"), [])
        }
        completed = {
            norm_key(value)
            for value in json_loads(getattr(row, "html_completed_fields_json"), [])
        }
        history_texts = {
            norm_key(value)
            for value in json_loads(getattr(row, "history_texts_json"), [])
        }
        probs = proba_by_id.get(row_id, np.zeros(len(label_encoder.classes_)))
        order = np.argsort(-probs) if len(probs) else np.array([], dtype=int)
        out[row_id] = {
            "task_raw": as_text(getattr(row, "task_raw")),
            "task_clean": as_text(getattr(row, "task_clean")),
            "task_norm": norm_key(getattr(row, "task_clean")),
            "task_tokens": task_tokens,
            "history_last_text_key": norm_key(getattr(row, "history_last_text")),
            "panel_names": panel_names,
            "panel_labels": panel_labels,
            "completed": completed,
            "history_texts": history_texts,
            "tag_proba": probs,
            "tag_order": order,
            "top_tag": label_encoder.classes_[order[0]] if len(order) else "",
            "top_tag_prob": float(probs[order[0]]) if len(order) else 0.0,
            "tag_margin": (
                float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else 0.0
            ),
        }
    return out


def build_rank_features(
    cand_df: pd.DataFrame,
    row_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    tag_proba: np.ndarray,
    category_maps: dict[str, dict[str, int]],
    site_freq: dict[str, float],
    priors: dict[str, Any],
) -> pd.DataFrame:
    row_ctx = build_row_context(row_df, label_encoder, tag_proba)
    label_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    feature_rows: list[dict[str, float | int]] = []

    for cand in cand_df.itertuples(index=False):
        row_id = as_text(getattr(cand, "id"))
        ctx = row_ctx[row_id]
        candidate_tag = as_text(getattr(cand, "candidate_tag"))
        tag_idx = label_to_idx.get(candidate_tag)
        probs = ctx["tag_proba"]
        order = ctx["tag_order"]
        if tag_idx is None or len(probs) == 0:
            tag_prob = 0.0
            tag_rank = len(label_encoder.classes_) + 1
        else:
            tag_prob = float(probs[tag_idx])
            where = np.where(order == tag_idx)[0]
            tag_rank = (
                int(where[0] + 1) if len(where) else len(label_encoder.classes_) + 1
            )

        task_tokens = ctx["task_tokens"]
        text_tokens = {
            token.lower()
            for token in json_loads(getattr(cand, "candidate_text_tokens_json"), [])
        }
        attrs_tokens = {
            token.lower()
            for token in json_loads(getattr(cand, "candidate_attrs_tokens_json"), [])
        }
        field_key = norm_key(getattr(cand, "candidate_field_key"))
        field_tokens = set(tokenize(field_key))
        cand_tokens = text_tokens | attrs_tokens | field_tokens | {candidate_tag}
        inter_text = len(task_tokens & text_tokens)
        inter_attrs = len(task_tokens & attrs_tokens)
        inter_field = len(task_tokens & field_tokens)
        inter_all = len(task_tokens & cand_tokens)

        name_key = norm_key(getattr(cand, "candidate_name"))
        label_key = norm_key(getattr(cand, "candidate_label"))
        placeholder_key = norm_key(getattr(cand, "candidate_placeholder"))
        keys = {key for key in [field_key, name_key, label_key, placeholder_key] if key}

        site = as_text(getattr(cand, "site_key"))
        global_prior = float(priors["global_prior"].get(candidate_tag, 0.0))
        site_prior = float(
            priors["site_prior"].get((site, candidate_tag), global_prior)
        )

        features = {
            "candidate_idx": int(getattr(cand, "candidate_idx")),
            "candidate_tag_idx": encode_value(
                candidate_tag, category_maps["candidate_tag"]
            ),
            "candidate_role_idx": encode_value(
                getattr(cand, "candidate_role"), category_maps["candidate_role"]
            ),
            "candidate_type_idx": encode_value(
                getattr(cand, "candidate_type"), category_maps["candidate_type"]
            ),
            "candidate_op_idx": encode_value(
                getattr(cand, "candidate_predicted_op"),
                category_maps["candidate_predicted_op"],
            ),
            "site_idx": encode_value(site, category_maps["site_key"]),
            "site_freq": float(site_freq.get(site, 0.0)),
            "html_type_idx": encode_value(
                getattr(cand, "html_type"), category_maps["html_type"]
            ),
            "is_workflow": int(getattr(cand, "is_workflow")),
            "history_last_op_idx": encode_value(
                getattr(cand, "history_last_op"), category_maps["history_last_op"]
            ),
            "history_step_count": int(getattr(cand, "history_step_count")),
            "html_current_step": int(getattr(cand, "html_current_step")),
            "html_total_steps": int(getattr(cand, "html_total_steps")),
            "html_progress": safe_ratio(
                int(getattr(cand, "html_current_step")),
                int(getattr(cand, "html_total_steps")),
            ),
            "candidate_tag_count_in_row": int(
                getattr(cand, "candidate_tag_count_in_row")
            ),
            "candidate_tag_frac_in_row": float(
                getattr(cand, "candidate_tag_frac_in_row")
            ),
            "candidate_tag_is_unique": int(
                int(getattr(cand, "candidate_tag_count_in_row")) == 1
            ),
            "candidate_text_empty": int(getattr(cand, "candidate_text_empty")),
            "candidate_attrs_empty": int(getattr(cand, "candidate_attrs_empty")),
            "candidate_options_count": int(getattr(cand, "candidate_options_count")),
            "candidate_token_count": len(cand_tokens),
            "task_token_count": len(task_tokens),
            "inter_task_text": inter_text,
            "inter_task_attrs": inter_attrs,
            "inter_task_field": inter_field,
            "inter_task_all": inter_all,
            "cov_task_cand": safe_ratio(inter_all, len(task_tokens)),
            "cov_cand_task": safe_ratio(inter_all, len(cand_tokens)),
            "text_cov_task": safe_ratio(inter_text, len(task_tokens)),
            "attrs_cov_task": safe_ratio(inter_attrs, len(task_tokens)),
            "field_cov_task": safe_ratio(inter_field, len(task_tokens)),
            "field_key_in_task": int(bool(field_key and field_key in ctx["task_norm"])),
            "name_in_task": int(bool(name_key and name_key in ctx["task_norm"])),
            "label_in_task": int(bool(label_key and label_key in ctx["task_norm"])),
            "placeholder_in_task": int(
                bool(placeholder_key and placeholder_key in ctx["task_norm"])
            ),
            "history_last_field_match": int(
                bool(field_key and field_key == ctx["history_last_text_key"])
            ),
            "in_history_texts": int(bool(keys & ctx["history_texts"])),
            "in_html_completed": int(bool(keys & ctx["completed"])),
            "in_workflow_panel": int(
                bool(keys & (ctx["panel_names"] | ctx["panel_labels"]))
            ),
            "tag_prob_for_candidate": tag_prob,
            "tag_rank_for_candidate": tag_rank,
            "tag_is_top1": int(candidate_tag == ctx["top_tag"]),
            "tag_is_top3": int(tag_rank <= 3),
            "tag_top1_prob": ctx["top_tag_prob"],
            "tag_margin": ctx["tag_margin"],
            "global_tag_prior": global_prior,
            "site_tag_prior": site_prior,
            "click_tag_hint": int(candidate_tag in CLICK_TAGS),
            "input_click_type_hint": int(
                candidate_tag == "input"
                and as_text(getattr(cand, "candidate_type")).lower()
                in CLICK_INPUT_TYPES
            ),
        }
        feature_rows.append(features)
    return pd.DataFrame(feature_rows).fillna(0)


CATEGORICAL_FEATURES = [
    "candidate_tag_idx",
    "candidate_role_idx",
    "candidate_type_idx",
    "candidate_op_idx",
    "site_idx",
    "html_type_idx",
    "history_last_op_idx",
]


def group_sizes(cand_df: pd.DataFrame) -> list[int]:
    return cand_df.groupby("id", sort=False).size().astype(int).tolist()


def fit_ranker(
    X_trn: pd.DataFrame,
    y_trn: np.ndarray,
    g_trn: list[int],
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    g_val: list[int] | None = None,
    config: dict[str, Any] | None = None,
    n_estimators: int = 1800,
) -> lgb.LGBMRanker:
    config = config or {}
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=config.get("learning_rate", 0.035),
        num_leaves=config.get("num_leaves", 63),
        min_child_samples=config.get("min_child_samples", 20),
        subsample=config.get("subsample", 0.9),
        colsample_bytree=config.get("colsample_bytree", 0.9),
        reg_lambda=1.0,
        label_gain=[0, 1],
        random_state=config.get("random_state", RNG_SEED),
        verbose=-1,
    )
    fit_kwargs: dict[str, Any] = {
        "X": X_trn,
        "y": y_trn,
        "group": g_trn,
        "categorical_feature": [
            col for col in CATEGORICAL_FEATURES if col in X_trn.columns
        ],
    }
    callbacks = []
    if X_val is not None and y_val is not None and g_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [g_val]
        fit_kwargs["eval_at"] = [1, 3, 5]
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(100)]
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    ranker.fit(**fit_kwargs)
    return ranker


def normalize_for_match(value: Any) -> str:
    text = norm_key(value)
    return f" {text} "


def parse_options_json(value: Any) -> list[str]:
    parsed = json_loads(value, [])
    return parsed if isinstance(parsed, list) else []


def predict_op(cand: pd.Series | dict[str, Any]) -> str:
    getter = (
        cand.get
        if isinstance(cand, dict)
        else lambda key, default="": getattr(cand, key, default)
    )
    tag = as_text(getter("candidate_tag", "")).lower()
    input_type = as_text(getter("candidate_type", "")).lower()
    predicted = as_text(getter("candidate_predicted_op", "")).upper()
    if predicted in {"CLICK", "TYPE", "SELECT"}:
        return predicted
    if tag == "select":
        return "SELECT"
    if tag == "textarea":
        return "TYPE"
    if tag == "input":
        return "CLICK" if input_type in CLICK_INPUT_TYPES else "TYPE"
    return "CLICK"


def extract_select_value(task: str, cand: dict[str, Any]) -> str:
    options = parse_options_json(cand.get("candidate_options_json", "[]"))
    if not options:
        return fallback_non_click_value(task, cand, "SELECT")
    task_norm = normalize_for_match(task)
    matches = []
    for option in options:
        option_norm = normalize_for_match(option).strip()
        if option_norm and f" {option_norm} " in task_norm:
            matches.append(option)
    if matches:
        return max(matches, key=len)
    for option in options:
        tokens = normalize_for_match(option).split()
        if tokens and all(f" {token} " in task_norm for token in tokens):
            matches.append(option)
    return max(matches, key=len) if matches else normalize_value(options[0])


def extract_after_field(task: str, field: str) -> str:
    field_text = as_text(field).strip()
    field_norm = norm_key(field_text)
    if not field_norm:
        return ""
    task_text = re.sub(r"^\s*Task\s*:\s*", "", as_text(task), flags=re.I).strip()
    alias_pattern = re.escape(field_norm).replace(r"\ ", r"\s+")
    pattern = re.compile(
        rf"\b{alias_pattern}\b\s*(?:is|as|to|for|with|=|:)?\s+(.+?)(?=,|;|\.(?:\s|$)|\s+\b(?:and|then|set|choose|select|mark|state|route|store|notify|queue|submit|click|open|schedule|due date|priority|mode|about)\b|$)",
        re.I,
    )
    match = pattern.search(task_text)
    if not match:
        return ""
    return cleanup_extracted_value(match.group(1))


def cleanup_extracted_value(value: Any) -> str:
    text = as_text(value).strip()
    text = re.sub(r"^\s*(?:is|as|to|for|with|=|:)\s+", "", text, flags=re.I)
    text = re.split(
        r"\s+\b(?:and|then|set|choose|select|mark|state|route|store|notify|queue|submit|click|open|schedule|due date|priority|mode|about)\b",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return text.strip(" \t\r\n\"'.,;:")


def decimal_to_compact(value: str) -> str:
    if not re.fullmatch(r"\d+\.\d+", value):
        return value
    return value.rstrip("0").rstrip(".")


def field_aliases(cand: dict[str, Any]) -> list[str]:
    values = [
        cand.get("candidate_label", ""),
        cand.get("candidate_name", ""),
        cand.get("candidate_field_key", ""),
        cand.get("candidate_text", ""),
    ]
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = cleanup_extracted_value(value)
        for alias in [raw, norm_key(raw)]:
            alias = alias.strip()
            key = alias.lower()
            if alias and key not in seen and key not in {"name", "text", "search"}:
                seen.add(key)
                aliases.append(alias)
    return aliases


def person_like_from_task(task: str, field_blob: str) -> str:
    field_tokens = set(field_blob.split())
    person_tokens = {
        "advisor",
        "borrower",
        "caller",
        "counsel",
        "donor",
        "employee",
        "inspector",
        "manager",
        "operator",
        "owner",
        "patron",
        "pilot",
        "producer",
        "recipient",
        "reviewer",
        "student",
        "technician",
    }
    if not field_tokens & person_tokens:
        return ""
    task_text = re.sub(r"^\s*Task\s*:\s*", "", as_text(task), flags=re.I).strip()
    triggers: list[str] = []
    if "advisor" in field_tokens:
        triggers += [r"choose\s+advisor", r"advisor"]
    if "student" in field_tokens:
        triggers += [r"add\s+student", r"student"]
    if "donor" in field_tokens:
        triggers += [r"for\s+donor", r"donor"]
    if "patron" in field_tokens:
        triggers += [r"for\s+patron", r"patron"]
    if "employee" in field_tokens or "recipient" in field_tokens:
        triggers += [r"for"]
    for role in ["borrower", "caller", "counsel", "inspector", "manager", "operator", "owner", "pilot", "producer", "reviewer", "technician"]:
        if role in field_tokens:
            triggers += [rf"assign(?:ed)?\s+{role}", rf"{role}"]
    name_pat = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
    for trigger in triggers:
        match = re.search(rf"(?i:\b{trigger}\s+){name_pat}\b", task_text)
        if match:
            return cleanup_extracted_value(match.group(1))
    return ""


def extract_type_value(task: str, cand: dict[str, Any]) -> str:
    field_values = [
        cand.get("candidate_label", ""),
        cand.get("candidate_name", ""),
        cand.get("candidate_field_key", ""),
        cand.get("candidate_placeholder", ""),
        cand.get("candidate_text", ""),
    ]
    field_blob = " ".join(norm_key(value) for value in field_values)

    person_value = person_like_from_task(task, field_blob)
    if person_value:
        return person_value

    if "email" in field_blob:
        match = EMAIL_RE.search(task)
        if match:
            return match.group(0)
    if any(
        key in field_blob
        for key in ["id", "code", "serial", "asset", "mission", "confirmation"]
    ):
        match = CODE_RE.search(task)
        if match:
            return match.group(0)
    if any(
        key in field_blob for key in ["amount", "price", "quantity", "number", "limit"]
    ):
        money = MONEY_RE.search(task)
        if money:
            return decimal_to_compact(money.group(1))
        match = DECIMAL_RE.search(task)
        if match:
            return decimal_to_compact(match.group(0))
    if "date" in field_blob:
        match = ISO_DATE_RE.search(task)
        if match:
            return match.group(0)

    for field in field_aliases(cand):
        value = extract_after_field(task, field)
        if value:
            return value

    for regex in [EMAIL_RE, ISO_DATE_RE, CODE_RE]:
        match = regex.search(task)
        if match:
            return match.group(0)
    return fallback_non_click_value(task, cand, "TYPE")


def fallback_non_click_value(task: str, cand: dict[str, Any], op: str) -> str:
    field_values = [
        cand.get("candidate_label", ""),
        cand.get("candidate_name", ""),
        cand.get("candidate_field_key", ""),
        cand.get("candidate_placeholder", ""),
        cand.get("candidate_text", ""),
    ]
    field_blob = " ".join(norm_key(value) for value in field_values)

    for regex in [EMAIL_RE, ISO_DATE_RE, CODE_RE]:
        match = regex.search(task)
        if match:
            return match.group(0)

    money = MONEY_RE.search(task)
    if money:
        return decimal_to_compact(money.group(1))

    if any(
        key in field_blob
        for key in ["amount", "count", "limit", "number", "price", "quantity"]
    ):
        match = DECIMAL_RE.search(task)
        if match:
            return decimal_to_compact(match.group(0))
        return "1"

    person_value = person_like_from_task(task, field_blob)
    if person_value:
        return person_value

    for value in field_values:
        text = normalize_value(value)
        if not text:
            continue
        low = text.lower()
        if low in {"yyyy-mm-dd", "name", "text", "search"} or "xxxxx" in low:
            continue
        return text

    task_tokens = [
        token
        for token in TOKEN_RE.findall(task)
        if token.lower() not in {"task", "click", "type", "select"}
    ]
    if task_tokens:
        return task_tokens[-1]
    return f"UNKNOWN_{op}_VALUE"


def predict_value(task: str, cand: dict[str, Any], op: str) -> str:
    if op == "CLICK":
        return ""
    if op == "SELECT":
        return extract_select_value(task, cand) or fallback_non_click_value(
            task, cand, op
        )
    if op == "TYPE":
        return extract_type_value(task, cand) or fallback_non_click_value(
            task, cand, op
        )
    return ""


def normalize_value(value: Any) -> str:
    return as_text(value).strip()


def predict_from_scores(
    cand_df: pd.DataFrame,
    row_df: pd.DataFrame,
    scores: np.ndarray,
    group_sizes_: list[int],
) -> pd.DataFrame:
    row_by_id = row_df.set_index("id").to_dict("index")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for size in group_sizes_:
        group = cand_df.iloc[cursor : cursor + size].reset_index(drop=True)
        group_scores = scores[cursor : cursor + size]
        cursor += size
        best_pos = int(np.argmax(group_scores))
        pred_cand = group.iloc[best_pos].to_dict()
        row_id = as_text(pred_cand["id"])
        row = row_by_id.get(row_id, {})
        task = as_text(row.get("task_raw") or row.get("task_clean"))
        op = predict_op(pred_cand)
        value = predict_value(task, pred_cand, op)
        if op in {"TYPE", "SELECT"} and not normalize_value(value):
            value = f"UNKNOWN_{op}_VALUE"
        pred_id = as_text(pred_cand.get("candidate_id"))

        true_target = as_text(row.get("target_id"))
        true_pos = -1
        if true_target:
            matches = np.where(
                group["candidate_id"].astype(str).to_numpy() == true_target
            )[0]
            true_pos = int(matches[0]) if len(matches) else -1
        order = np.argsort(-group_scores)
        true_rank = int(np.where(order == true_pos)[0][0]) + 1 if true_pos >= 0 else 999

        rows.append(
            {
                "id": row_id,
                "op": op,
                "target_id": pred_id,
                "value": value,
                "html_type": as_text(row.get("html_type")),
                "true_target_id": true_target,
                "true_op": as_text(row.get("op")),
                "true_value": normalize_value(row.get("value")),
                "target_rank": true_rank,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(pred: pd.DataFrame) -> dict[str, float]:
    has_labels = pred["true_target_id"].astype(str).str.len().sum() > 0
    if not has_labels:
        return {}
    pred = pred.copy()
    pred["target_match"] = pred["target_id"] == pred["true_target_id"]
    pred["op_match"] = pred["op"] == pred["true_op"]
    pred["value_match"] = pred["value"].map(normalize_value) == pred["true_value"].map(
        normalize_value
    )
    pred["all_match"] = pred["target_match"] & pred["op_match"] & pred["value_match"]
    metrics = {
        "target_id_acc": float(pred["target_match"].mean()),
        "target_top3_acc": float((pred["target_rank"] <= 3).mean()),
        "target_top5_acc": float((pred["target_rank"] <= 5).mean()),
        "op_acc": float(pred["op_match"].mean()),
        "value_acc": float(pred["value_match"].mean()),
        "all_match_acc": float(pred["all_match"].mean()),
    }
    for html_type in ["workflow", "raw_web"]:
        part = pred[pred["html_type"] == html_type]
        if len(part):
            metrics[f"{html_type}_target_id_acc"] = float(part["target_match"].mean())
            metrics[f"{html_type}_all_match_acc"] = float(part["all_match"].mean())
    return metrics


def enforce_non_click_values(
    pred: pd.DataFrame, row_df: pd.DataFrame, cand_df: pd.DataFrame
) -> pd.DataFrame:
    pred = pred.copy()
    row_by_id = row_df.set_index("id").to_dict("index")
    cand_by_key = {
        (as_text(row.id), as_text(row.candidate_id)): row._asdict()
        for row in cand_df.itertuples(index=False)
    }
    blank_mask = pred["op"].isin(["TYPE", "SELECT"]) & pred["value"].fillna("").astype(
        str
    ).str.strip().eq("")
    for idx, row in pred.loc[blank_mask].iterrows():
        op = as_text(row["op"])
        row_id = as_text(row["id"])
        target_id = as_text(row["target_id"])
        row_ctx = row_by_id.get(row_id, {})
        task = as_text(row_ctx.get("task_raw") or row_ctx.get("task_clean"))
        cand = cand_by_key.get((row_id, target_id), {})
        value = predict_value(task, cand, op) if cand else ""
        if not normalize_value(value):
            value = (
                fallback_non_click_value(task, cand, op)
                if cand
                else f"UNKNOWN_{op}_VALUE"
            )
        if not normalize_value(value):
            value = f"UNKNOWN_{op}_VALUE"
        pred.at[idx, "value"] = value
    return pred


def save_submission(
    pred: pd.DataFrame, row_df: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    sub = row_df[["id"]].merge(
        pred[["id", "op", "target_id", "value"]], on="id", how="left"
    )
    sub["op"] = sub["op"].fillna("CLICK")
    sub["target_id"] = sub["target_id"].fillna("")
    sub["value"] = sub["value"].fillna("")
    sub.loc[sub["op"] == "CLICK", "value"] = ""
    non_click_blank = sub["op"].isin(["TYPE", "SELECT"]) & sub["value"].astype(
        str
    ).str.strip().eq("")
    sub.loc[non_click_blank, "value"] = sub.loc[non_click_blank, "op"].map(
        lambda op: f"UNKNOWN_{op}_VALUE"
    )
    sub[["id", "op", "target_id", "value"]].to_csv(
        out_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    return sub[["id", "op", "target_id", "value"]]


def save_workflow_only_submission(
    submission: pd.DataFrame, row_df: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    html_type = row_df[["id", "html_type"]]
    out = submission.merge(html_type, on="id", how="left")
    raw_mask = out["html_type"] != "workflow"
    out.loc[raw_mask, "op"] = "GARBAGE_OP"
    out.loc[raw_mask, "target_id"] = "GARBAGE_TARGET_ID"
    out.loc[raw_mask, "value"] = "GARBAGE_VALUE"
    out = out[["id", "op", "target_id", "value"]]
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    return out


def feature_importance(
    models: list[lgb.LGBMRanker], feature_names: list[str]
) -> pd.DataFrame:
    parts = []
    for idx, model in enumerate(models):
        parts.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    f"model_{idx}_gain": model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                    f"model_{idx}_split": model.booster_.feature_importance(
                        importance_type="split"
                    ),
                }
            )
        )
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="feature", how="left")
    gain_cols = [col for col in out.columns if col.endswith("_gain")]
    split_cols = [col for col in out.columns if col.endswith("_split")]
    out["gain"] = out[gain_cols].mean(axis=1)
    out["split"] = out[split_cols].mean(axis=1)
    return out.sort_values("gain", ascending=False)


def run_training(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TRAIN_ROW_PATH:", TRAIN_ROW_PATH)
    print("TRAIN_CAND_PATH:", TRAIN_CAND_PATH)

    row_df, cand_df = load_train_data()
    train_ids, val_ids = make_id_split(row_df)
    trn_rows = subset_rows(row_df, train_ids)
    val_rows = subset_rows(row_df, val_ids)
    trn_cands = subset_candidates(cand_df, train_ids)
    val_cands = subset_candidates(cand_df, val_ids)
    print("train rows:", len(trn_rows), "valid rows:", len(val_rows))
    print("train candidates:", len(trn_cands), "valid candidates:", len(val_cands))

    print("[fit] tag model")
    tag_model, tag_encoder, tag_bundle, trn_tag_proba, val_tag_proba, tag_metric = (
        fit_tag_model(trn_rows, val_rows, trn_cands)
    )
    print("tag metrics:", tag_metric)

    print("[features] ranker")
    rank_category_maps = build_category_maps(trn_rows, trn_cands)
    site_freq = trn_rows["site_key"].value_counts(normalize=True).to_dict()
    priors = compute_priors(trn_rows)
    X_trn = build_rank_features(
        trn_cands,
        trn_rows,
        tag_encoder,
        trn_tag_proba,
        rank_category_maps,
        site_freq,
        priors,
    )
    X_val = build_rank_features(
        val_cands,
        val_rows,
        tag_encoder,
        val_tag_proba,
        rank_category_maps,
        site_freq,
        priors,
    )
    y_trn = trn_cands["label_is_target"].astype(int).to_numpy()
    y_val = val_cands["label_is_target"].astype(int).to_numpy()
    g_trn = group_sizes(trn_cands)
    g_val = group_sizes(val_cands)
    print("rank features:", X_trn.shape[1])

    print("[fit] ranker ensemble")
    rankers: list[lgb.LGBMRanker] = []
    val_score_parts: list[np.ndarray] = []
    model_summaries = []
    for config in RANKER_CONFIGS:
        print(" -", config["name"])
        ranker = fit_ranker(
            X_trn,
            y_trn,
            g_trn,
            X_val,
            y_val,
            g_val,
            config=config,
            n_estimators=args.n_estimators,
        )
        scores = ranker.predict(X_val, num_iteration=ranker.best_iteration_)
        pred_single = predict_from_scores(val_cands, val_rows, scores, g_val)
        single_metrics = evaluate_predictions(pred_single)
        model_summaries.append(
            {
                "name": config["name"],
                "best_iteration": int(ranker.best_iteration_ or 0),
                "target_id_acc": single_metrics.get("target_id_acc", 0.0),
                "all_match_acc": single_metrics.get("all_match_acc", 0.0),
            }
        )
        rankers.append(ranker)
        val_score_parts.append(scores)

    val_scores = np.mean(val_score_parts, axis=0)
    pred_val = predict_from_scores(val_cands, val_rows, val_scores, g_val)
    pred_val = enforce_non_click_values(pred_val, val_rows, val_cands)
    metrics = {**tag_metric, **evaluate_predictions(pred_val)}
    metrics["ensemble_size"] = len(rankers)
    metrics["rank_features"] = X_trn.shape[1]
    metrics["tag_best_iteration"] = int(tag_model.best_iteration_ or 0)
    print("[metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    pred_val.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(SCRIPT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    feature_importance(rankers, list(X_trn.columns)).to_csv(
        OUTPUT_DIR / "feature_importance_ranker.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    tag_importance = pd.DataFrame(
        {
            "feature": tag_bundle["feature_names"],
            "gain": tag_model.booster_.feature_importance(importance_type="gain"),
            "split": tag_model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    tag_importance.to_csv(
        OUTPUT_DIR / "feature_importance_tag.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print("[refit] full train")
    final_tag_model, final_tag_encoder, final_tag_bundle, full_tag_proba = (
        fit_final_tag_model(row_df, cand_df)
    )
    final_rank_category_maps = build_category_maps(row_df, cand_df)
    final_site_freq = row_df["site_key"].value_counts(normalize=True).to_dict()
    final_priors = compute_priors(row_df)
    X_all = build_rank_features(
        cand_df,
        row_df,
        final_tag_encoder,
        full_tag_proba,
        final_rank_category_maps,
        final_site_freq,
        final_priors,
    )
    y_all = cand_df["label_is_target"].astype(int).to_numpy()
    g_all = group_sizes(cand_df)

    final_rankers: list[lgb.LGBMRanker] = []
    final_iterations = []
    for config, holdout_ranker in zip(RANKER_CONFIGS, rankers):
        final_iters = int(max(250, (holdout_ranker.best_iteration_ or 500) * 1.15))
        final_iterations.append(final_iters)
        print(f" - {config['name']} final_iters={final_iters}")
        final_rankers.append(
            fit_ranker(X_all, y_all, g_all, config=config, n_estimators=final_iters)
        )

    bundle = {
        "tag_model": final_tag_model,
        "tag_encoder": final_tag_encoder,
        "tag_bundle": final_tag_bundle,
        "rankers": final_rankers,
        "rank_category_maps": final_rank_category_maps,
        "site_freq": final_site_freq,
        "priors": final_priors,
        "rank_feature_names": list(X_all.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
        "model_summaries": model_summaries,
        "final_iterations": final_iterations,
    }
    joblib.dump(bundle, OUTPUT_DIR / "lgbm_ranker_v6_bundle.joblib")
    joblib.dump(
        {
            "model": final_tag_model,
            "encoder": final_tag_encoder,
            "bundle": final_tag_bundle,
        },
        OUTPUT_DIR / "tag_model.joblib",
    )

    print("[predict] test")
    test_rows, test_cands, source_path = load_or_build_test_data()
    if source_path:
        print("test source:", source_path)
    test_tag_proba = predict_tag_proba(
        final_tag_model, final_tag_encoder, final_tag_bundle, test_rows
    )
    X_test = build_rank_features(
        test_cands,
        test_rows,
        final_tag_encoder,
        test_tag_proba,
        final_rank_category_maps,
        final_site_freq,
        final_priors,
    )
    g_test = group_sizes(test_cands)
    test_scores = np.mean([model.predict(X_test) for model in final_rankers], axis=0)
    pred_test = predict_from_scores(test_cands, test_rows, test_scores, g_test)
    pred_test = enforce_non_click_values(pred_test, test_rows, test_cands)
    pred_test.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    submission = save_submission(pred_test, test_rows, SUBMISSION_PATH)
    submission.to_csv(
        OUTPUT_DIR / "submission.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    workflow_only = save_workflow_only_submission(
        submission, test_rows, WORKFLOW_ONLY_SUBMISSION_PATH
    )
    workflow_only.to_csv(
        OUTPUT_DIR / "submission_workflow_only_rawweb_garbage.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    print("submission rows:", len(submission))
    print(
        "workflow-only garbage rows:", int((test_rows["html_type"] != "workflow").sum())
    )
    print("saved:", SCRIPT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v6 tag-assisted LightGBM ranker training."
    )
    parser.add_argument("--n-estimators", type=int, default=1800)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
