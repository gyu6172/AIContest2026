from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


RNG_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "HyoungGyu" / "preprocessing").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/ and HyoungGyu/preprocessing/.")


ROOT = find_project_root(SCRIPT_DIR)
DATA_DIR = ROOT / "data"
PREPROCESS_DIR = ROOT / "HyoungGyu" / "preprocessing"

TRAIN_PATH = PREPROCESS_DIR / "preprocessed_train_v5.csv"
TEST_PATH = PREPROCESS_DIR / "preprocessed_test_v5.csv"
RAW_TRAIN_PATH = DATA_DIR / "train.csv"
RAW_TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_DIR = SCRIPT_DIR / "artifacts"


DOMAIN_STOP = {"task", "step", "enter", "action", "element"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP
TOKEN_RE = re.compile(r"[A-Za-z0-9/._-]+")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
ATTR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*)=([^|]+?)(?=\s*\||$)", re.I)
OPT_RE = re.compile(r"\boptions=([^|]+?)(?=\s*\||$)", re.I)
TYPE_ATTR_RE = re.compile(r"\btype=([^|]+?)(?=\s*\||$)", re.I)

TAG_VOCAB = [
    "input",
    "textarea",
    "select",
    "button",
    "a",
    "label",
    "div",
    "li",
    "span",
    "svg",
    "img",
    "td",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "section",
    "option",
]
INPUT_TYPE_VOCAB = [
    "text",
    "date",
    "email",
    "number",
    "checkbox",
    "radio",
    "button",
    "submit",
    "password",
    "tel",
    "url",
    "time",
    "search",
    "range",
    "image",
]
OP_VOCAB = ["CLICK", "TYPE", "SELECT"]

CLICK_TAGS = {
    "a",
    "button",
    "div",
    "span",
    "li",
    "img",
    "svg",
    "label",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "td",
}
TYPE_INPUT_TYPES = {"text", "date", "email", "number", "password", "tel", "url", "time", "search"}
CLICK_INPUT_TYPES = {"checkbox", "radio", "button", "submit", "image", "range"}

TYPE_KEYWORDS = [
    "type",
    "enter",
    "input",
    "fill",
    "write",
    "set",
    "add",
    "assign",
    "create",
    "schedule",
    "update",
]
SELECT_KEYWORDS = ["select", "choose", "pick", "dropdown", "mark", "set"]
CLICK_KEYWORDS = ["click", "open", "submit", "save", "publish", "send", "queue", "dispatch", "approve", "view"]


HTML_COMPLETED_RE = re.compile(r"Completed:\s*([^<\n]+)", re.I)
HTML_STEP_RE = re.compile(r"current step\s+(\d+)\s+of\s+(\d+)", re.I)
PANEL_RE = re.compile(r'<section[^>]*aria-label="current workflow panel"[^>]*>(.*?)</section>', re.I | re.S)
NAME_ATTR_HTML_RE = re.compile(r'\bname="([^"]+)"', re.I)
H1_RE = re.compile(r"<h1[^>]*>([^<]+)</h1>", re.I)
WORKFLOW_CTX_RE = re.compile(r'<aside[^>]*class="workflow-context"[^>]*>(.*?)</aside>', re.I | re.S)
TAG_STRIP_RE = re.compile(r"<[^>]+>")


BASE_FEAT_NAMES = [
    "inter_RC",
    "inter_TC",
    "inter_HC",
    "sz_R",
    "sz_T",
    "sz_H",
    "sz_C",
    "cov_R_C",
    "cov_T_C",
    "cov_H_C",
    "cov_C_R",
    "jac_RC",
    "jac_TC",
    "dice_RC",
    "cov_R_C_idf",
    "cov_T_C_idf",
    "idf_inter_RC",
    "label_match_T",
    "name_match_T",
    "placeholder_match_T",
    "label_match_R",
    "name_match_R",
    "placeholder_match_R",
    "options_in_task",
    "n_options",
    "text_in_history",
    "label_in_history",
    "pos",
    "n_candidates",
    "pos_norm",
    "n_history_steps",
    "tag_idx",
    "input_type_idx",
    "cand_op_idx",
    "task_kw_click",
    "task_kw_type",
    "task_kw_select",
    "cand_op_kw_score",
    "has_label",
    "has_name",
    "has_placeholder",
    "has_options",
    "tag_count_in_row",
    "op_count_in_row",
    "is_unique_tag",
    "is_unique_op",
    "site_token_freq",
    "candidate_len",
    "html_len_log",
    "is_completed",
    "n_completed",
    "html_current_step",
    "html_total_steps",
    "html_step_remaining",
    "in_workflow_panel",
    "n_panel_names",
    "h1_match_C",
    "workflow_ctx_match_C",
    "global_tag_prior",
    "global_input_type_prior",
    "global_label_prior",
    "site_tag_prior",
    "site_label_prior",
    "site_op_prior",
]

ROW_NORM_SOURCE = [
    "cov_R_C_idf",
    "cov_T_C_idf",
    "idf_inter_RC",
    "jac_RC",
    "jac_TC",
    "label_match_T",
    "name_match_T",
    "placeholder_match_T",
    "candidate_len",
    "site_label_prior",
]
ROW_NORM_FEAT_NAMES = [
    f"{name}_{suffix}"
    for name in ROW_NORM_SOURCE
    for suffix in ("rank", "z", "fmax")
]
FEAT_NAMES = BASE_FEAT_NAMES + ROW_NORM_FEAT_NAMES
CATEGORICAL = ["tag_idx", "input_type_idx", "cand_op_idx"]

# The wider feature set above is useful for experiments, but the current holdout
# is strongest with the fixed v1 ranker surface plus the improved op/value rules.
MODEL_FEAT_NAMES = [
    "inter_RC",
    "inter_TC",
    "sz_R",
    "sz_T",
    "sz_C",
    "cov_R_C",
    "cov_T_C",
    "cov_C_R",
    "jac_RC",
    "jac_TC",
    "dice_RC",
    "cov_R_C_idf",
    "cov_T_C_idf",
    "idf_inter_RC",
    "label_match_T",
    "name_match_T",
    "placeholder_match_T",
    "options_in_task",
    "n_options",
    "text_in_history",
    "label_in_history",
    "pos",
    "n_candidates",
    "pos_norm",
    "n_history_steps",
    "tag_idx",
    "input_type_idx",
    "site_token_freq",
    "candidate_len",
    "html_len_log",
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def safe_json_loads(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        value = json.loads(x)
        return value if isinstance(value, list) else []
    except Exception:
        return []


def tokenize(s) -> list[str]:
    if not isinstance(s, str):
        s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return [
        token
        for token in (m.lower() for m in TOKEN_RE.findall(s))
        if token not in STOPWORDS and token
    ]


def simple_words(s) -> list[str]:
    if not isinstance(s, str):
        return []
    return [m.group(0).lower() for m in WORD_RE.finditer(split_field_text(s))]


def compute_idf(texts: list[str]) -> dict[str, float]:
    n = len(texts)
    df = Counter()
    for text in texts:
        for token in set(tokenize(text)):
            df[token] += 1
    return {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def get_attr(attrs, key: str) -> str:
    if not isinstance(attrs, str):
        return ""
    match = re.search(rf"\b{re.escape(key)}=([^|]+?)(?=\s*\||$)", attrs, re.I)
    return match.group(1).strip() if match else ""


def parse_options(attrs) -> list[str]:
    if not isinstance(attrs, str):
        return []
    match = OPT_RE.search(attrs)
    if not match:
        return []
    return [option.strip() for option in match.group(1).split("/") if option.strip()]


def candidate_text(cand: dict) -> str:
    return f"{cand.get('text') or ''} {cand.get('attrs') or ''}"


def split_field_text(value: str) -> str:
    value = (value or "").replace("_", " ")
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[^A-Za-z0-9]+", " ", value).strip()


def norm_key(value: str) -> str:
    words = simple_words(value)
    return " ".join(words)


def candidate_label_key(cand: dict) -> str:
    attrs = cand.get("attrs") or ""
    for value in [get_attr(attrs, "name"), get_attr(attrs, "label"), cand.get("text") or ""]:
        key = norm_key(value)
        if key:
            return key
    return ""


def vocab_index(value: str, vocab: list[str]) -> int:
    value = (value or "").lower()
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab)


def input_type_value(attrs: str) -> str:
    match = TYPE_ATTR_RE.search(attrs or "")
    return match.group(1).strip().lower() if match else ""


def input_type_index(attrs: str) -> int:
    return vocab_index(input_type_value(attrs), INPUT_TYPE_VOCAB)


def predict_op(cand: dict) -> str:
    tag = (cand.get("tag") or "").lower()
    attrs = cand.get("attrs") or ""
    input_type = input_type_value(attrs)
    role = get_attr(attrs, "role").lower()

    if tag == "select":
        return "SELECT"
    if tag == "textarea":
        return "TYPE"
    if tag == "input":
        if input_type in CLICK_INPUT_TYPES or role == "button":
            return "CLICK"
        return "TYPE"
    if tag in {"button", "a", "link"}:
        return "CLICK"
    return "CLICK"


def parse_html_context(html: str):
    if not isinstance(html, str):
        return set(), 0, 0, set(), set(), set()

    completed = set()
    for match in HTML_COMPLETED_RE.finditer(html):
        for item in match.group(1).split(","):
            item_key = norm_key(item)
            if item_key:
                completed.add(item_key)

    step_match = HTML_STEP_RE.search(html)
    current_step = int(step_match.group(1)) if step_match else 0
    total_steps = int(step_match.group(2)) if step_match else 0

    panel_names = set()
    panel_match = PANEL_RE.search(html)
    if panel_match:
        panel_names = {norm_key(name) for name in NAME_ATTR_HTML_RE.findall(panel_match.group(1))}
        panel_names.discard("")

    h1_tokens = set()
    for match in H1_RE.finditer(html):
        h1_tokens |= set(tokenize(match.group(1)))

    workflow_tokens = set()
    for match in WORKFLOW_CTX_RE.finditer(html):
        text = TAG_STRIP_RE.sub(" ", match.group(1))
        workflow_tokens |= set(tokenize(text))

    return completed, current_step, total_steps, panel_names, h1_tokens, workflow_tokens


def candidate_completed(cand: dict, completed: set[str]) -> float:
    if not completed:
        return 0.0
    attrs = cand.get("attrs") or ""
    keys = [
        norm_key(cand.get("text") or ""),
        norm_key(get_attr(attrs, "label")),
        norm_key(get_attr(attrs, "name")),
    ]
    return 1.0 if any(key and key in completed for key in keys) else 0.0


def count_keywords(text: str, keywords: list[str]) -> int:
    text = (text or "").lower()
    return sum(1 for keyword in keywords if keyword in text)


def smoothed_prior(pos: int, total: int, base_rate: float, alpha: float = 3.0) -> float:
    return (pos + alpha * base_rate) / (total + alpha)


def compute_prior_stats(df: pd.DataFrame) -> dict:
    counters = {
        "global_tag_pos": Counter(),
        "global_tag_total": Counter(),
        "global_input_type_pos": Counter(),
        "global_input_type_total": Counter(),
        "global_label_pos": Counter(),
        "global_label_total": Counter(),
        "site_tag_pos": Counter(),
        "site_tag_total": Counter(),
        "site_label_pos": Counter(),
        "site_label_total": Counter(),
        "site_op_pos": Counter(),
        "site_op_total": Counter(),
    }

    total_candidates = 0
    total_positive = 0
    for _, row in df.iterrows():
        site = row.get("site_token", "")
        target_id = row.get("target_id", "")
        for cand in row.get("candidate_list", []):
            if not isinstance(cand, dict):
                continue
            total_candidates += 1
            is_target = cand.get("candidate_id") == target_id
            total_positive += int(is_target)

            tag = (cand.get("tag") or "").lower()
            input_type = input_type_value(cand.get("attrs") or "")
            label = candidate_label_key(cand)
            op = predict_op(cand)
            keys = [
                ("global_tag", tag),
                ("global_input_type", input_type),
                ("global_label", label),
                ("site_tag", (site, tag)),
                ("site_label", (site, label)),
                ("site_op", (site, op)),
            ]
            for name, key in keys:
                counters[f"{name}_total"][key] += 1
                if is_target:
                    counters[f"{name}_pos"][key] += 1

    base_rate = total_positive / total_candidates if total_candidates else 1.0 / 15.0
    counters["base_rate"] = base_rate
    return counters


def prior_value(stats: dict, name: str, key) -> float:
    base_rate = stats.get("base_rate", 1.0 / 15.0)
    pos = stats[f"{name}_pos"].get(key, 0)
    total = stats[f"{name}_total"].get(key, 0)
    return smoothed_prior(pos, total, base_rate)


def add_row_norm_features(row_dicts: list[dict]) -> None:
    n = len(row_dicts)
    for name in ROW_NORM_SOURCE:
        vals = np.asarray([float(d.get(name, 0.0)) for d in row_dicts], dtype=float)
        mean = float(vals.mean()) if n else 0.0
        std = float(vals.std()) if n else 1.0
        if std < 1e-9:
            std = 1.0
        max_abs = float(np.max(np.abs(vals))) if n else 0.0
        denom = max_abs if max_abs > 1e-9 else 1.0
        for idx, d in enumerate(row_dicts):
            rank = int(np.sum(vals > vals[idx]))
            d[f"{name}_rank"] = rank / max(1, n - 1)
            d[f"{name}_z"] = (vals[idx] - mean) / std
            d[f"{name}_fmax"] = vals[idx] / denom


def build_rank_dataset(
    df_part: pd.DataFrame,
    idf: dict[str, float],
    site_freq: dict[str, float],
    prior_stats: dict,
):
    X_rows = []
    y_rows = []
    group_sizes = []
    meta = []

    for _, row in df_part.iterrows():
        cands = row.get("candidate_list", [])
        if not cands:
            continue

        task = row.get("task", "")
        history = row.get("history", "")
        html = row.get("cleaned_html", "")
        raw_html = row.get("cleaned_html_raw", html)
        task_l = task if isinstance(task, str) else ""
        history_l = history if isinstance(history, str) else ""
        raw_task = row.get("task_raw", task_l)

        T = set(tokenize(task_l))
        H = set(tokenize(history_l))
        R = T - H
        R_idf_sum = sum(idf.get(t, 1.0) for t in R) if R else 0.0
        T_idf_sum = sum(idf.get(t, 1.0) for t in T) if T else 0.0
        n_steps = len(re.findall(r"\b\d+\s+\w+", history_l)) if isinstance(history_l, str) else 0
        n_cands = len(cands)
        html_len_log = math.log1p(len(html) if isinstance(html, str) else 0)
        site_token = row.get("site_token", "")
        site_token_freq = site_freq.get(site_token, 0.0)

        completed, current_step, total_steps, panel_names, h1_tokens, workflow_tokens = parse_html_context(raw_html)
        task_kw_click = count_keywords(raw_task, CLICK_KEYWORDS)
        task_kw_type = count_keywords(raw_task, TYPE_KEYWORDS)
        task_kw_select = count_keywords(raw_task, SELECT_KEYWORDS)
        op_kw_scores = {"CLICK": task_kw_click, "TYPE": task_kw_type, "SELECT": task_kw_select}

        raw_cands = row.get("raw_candidate_list") or cands
        raw_by_id = {c.get("candidate_id"): c for c in raw_cands if isinstance(c, dict)}
        raw_candidates = [raw_by_id.get(c.get("candidate_id"), c) for c in cands if isinstance(c, dict)]

        row_tag_counts = Counter((c.get("tag") or "").lower() for c in cands if isinstance(c, dict))
        row_op_counts = Counter(predict_op(c) for c in cands if isinstance(c, dict))

        cand_ids = []
        row_feat_dicts = []
        for pos, cand in enumerate(cands):
            if not isinstance(cand, dict):
                continue

            cand_id = cand.get("candidate_id") or ""
            attrs = cand.get("attrs") or ""
            text = cand.get("text") or ""
            tag = (cand.get("tag") or "").lower()
            input_type = input_type_value(attrs)
            cand_op = predict_op(cand)
            C = set(tokenize(candidate_text(cand)))

            inter_RC = len(R & C)
            inter_TC = len(T & C)
            inter_HC = len(H & C)
            union_RC = R | C
            union_TC = T | C
            idf_inter_RC = sum(idf.get(t, 1.0) for t in (R & C))
            idf_inter_TC = sum(idf.get(t, 1.0) for t in (T & C))

            label = get_attr(attrs, "label")
            name = get_attr(attrs, "name")
            placeholder = get_attr(attrs, "placeholder")
            label_tokens = set(tokenize(label))
            name_tokens = set(tokenize(name.replace("_", " ")))
            placeholder_tokens = set(tokenize(placeholder))
            options = parse_options(attrs)
            label_key = candidate_label_key(cand)

            cand_name_key = norm_key(name)
            tag_count = row_tag_counts[tag]
            op_count = row_op_counts[cand_op]
            html_step_remaining = max(0, total_steps - current_step) if total_steps else 0

            d = {
                "inter_RC": inter_RC,
                "inter_TC": inter_TC,
                "inter_HC": inter_HC,
                "sz_R": len(R),
                "sz_T": len(T),
                "sz_H": len(H),
                "sz_C": len(C),
                "cov_R_C": inter_RC / len(R) if R else 0.0,
                "cov_T_C": inter_TC / len(T) if T else 0.0,
                "cov_H_C": inter_HC / len(H) if H else 0.0,
                "cov_C_R": inter_RC / len(C) if C else 0.0,
                "jac_RC": inter_RC / len(union_RC) if union_RC else 0.0,
                "jac_TC": inter_TC / len(union_TC) if union_TC else 0.0,
                "dice_RC": 2 * inter_RC / (len(R) + len(C)) if (R or C) else 0.0,
                "cov_R_C_idf": idf_inter_RC / R_idf_sum if R_idf_sum > 0 else 0.0,
                "cov_T_C_idf": idf_inter_TC / T_idf_sum if T_idf_sum > 0 else 0.0,
                "idf_inter_RC": idf_inter_RC,
                "label_match_T": len(T & label_tokens) / len(label_tokens) if label_tokens else 0.0,
                "name_match_T": len(T & name_tokens) / len(name_tokens) if name_tokens else 0.0,
                "placeholder_match_T": len(T & placeholder_tokens) / len(placeholder_tokens) if placeholder_tokens else 0.0,
                "label_match_R": len(R & label_tokens) / len(label_tokens) if label_tokens else 0.0,
                "name_match_R": len(R & name_tokens) / len(name_tokens) if name_tokens else 0.0,
                "placeholder_match_R": len(R & placeholder_tokens) / len(placeholder_tokens) if placeholder_tokens else 0.0,
                "options_in_task": sum(1 for opt in options if opt.lower() in str(raw_task).lower()),
                "n_options": len(options),
                "text_in_history": 1.0 if text and text.lower() in history_l.lower() else 0.0,
                "label_in_history": 1.0 if label and label.lower() in history_l.lower() else 0.0,
                "pos": pos,
                "n_candidates": n_cands,
                "pos_norm": pos / max(1, n_cands - 1),
                "n_history_steps": n_steps,
                "tag_idx": vocab_index(tag, TAG_VOCAB),
                "input_type_idx": vocab_index(input_type, INPUT_TYPE_VOCAB),
                "cand_op_idx": vocab_index(cand_op, OP_VOCAB),
                "task_kw_click": task_kw_click,
                "task_kw_type": task_kw_type,
                "task_kw_select": task_kw_select,
                "cand_op_kw_score": op_kw_scores.get(cand_op, 0),
                "has_label": 1.0 if label else 0.0,
                "has_name": 1.0 if name else 0.0,
                "has_placeholder": 1.0 if placeholder else 0.0,
                "has_options": 1.0 if options else 0.0,
                "tag_count_in_row": tag_count,
                "op_count_in_row": op_count,
                "is_unique_tag": 1.0 if tag_count == 1 else 0.0,
                "is_unique_op": 1.0 if op_count == 1 else 0.0,
                "site_token_freq": site_token_freq,
                "candidate_len": len(candidate_text(cand)),
                "html_len_log": html_len_log,
                "is_completed": candidate_completed(cand, completed),
                "n_completed": len(completed),
                "html_current_step": current_step,
                "html_total_steps": total_steps,
                "html_step_remaining": html_step_remaining,
                "in_workflow_panel": 1.0 if cand_name_key and cand_name_key in panel_names else 0.0,
                "n_panel_names": len(panel_names),
                "h1_match_C": len(h1_tokens & C) / len(C) if C else 0.0,
                "workflow_ctx_match_C": len(workflow_tokens & C) / len(C) if C else 0.0,
                "global_tag_prior": prior_value(prior_stats, "global_tag", tag),
                "global_input_type_prior": prior_value(prior_stats, "global_input_type", input_type),
                "global_label_prior": prior_value(prior_stats, "global_label", label_key),
                "site_tag_prior": prior_value(prior_stats, "site_tag", (site_token, tag)),
                "site_label_prior": prior_value(prior_stats, "site_label", (site_token, label_key)),
                "site_op_prior": prior_value(prior_stats, "site_op", (site_token, cand_op)),
            }
            row_feat_dicts.append(d)
            cand_ids.append(cand_id)

        add_row_norm_features(row_feat_dicts)
        for d, cand_id in zip(row_feat_dicts, cand_ids):
            X_rows.append([float(d.get(name, 0.0)) for name in FEAT_NAMES])
            y_rows.append(1.0 if cand_id == row.get("target_id") else 0.0)

        group_sizes.append(len(cand_ids))
        meta.append(
            {
                "id": row["id"],
                "target_id": row.get("target_id", ""),
                "true_op": row.get("op", ""),
                "true_value": row.get("value", ""),
                "task": task_l,
                "task_raw": row.get("task_raw", task_l),
                "candidates": cands,
                "raw_candidates": raw_candidates,
                "raw_by_id": raw_by_id,
                "candidate_ids": cand_ids,
            }
        )

    X = pd.DataFrame(X_rows, columns=FEAT_NAMES)
    y = np.asarray(y_rows, dtype=float)
    groups = np.asarray(group_sizes, dtype=int)
    return X, y, groups, meta


GENERIC_FIELD_WORDS = {
    "id",
    "code",
    "number",
    "date",
    "name",
    "input",
    "field",
    "search",
    "q",
    "query",
    "text",
    "value",
    "the",
    "your",
    "for",
    "to",
    "from",
    "city",
    "airport",
    "location",
    "zip",
    "zipcode",
}
VALUE_TAIL_RE = re.compile(
    r"\s+(?:and|with|then|before|after|using|choose|select|set|mark|schedule|submit|publish|book|save|create|open|assign|notify|approve|dispatch|send|queue|add|for|from|to|at)\b.*$",
    re.I,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CODE_RE = re.compile(r"\b[A-Z]{2,6}-[A-Z0-9]{2,10}\b")
DECIMAL_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
MONEY_RE = re.compile(r"[$€£]\s*(\d+(?:\.\d+)?)")


def field_blob(cand: dict) -> str:
    raw = " ".join([cand.get("text") or "", cand.get("attrs") or ""])
    return f"{raw} {split_field_text(raw)}".lower()


def candidate_field_strings(cand: dict) -> list[str]:
    values = []
    attrs = cand.get("attrs") or ""
    for value in [cand.get("text") or "", get_attr(attrs, "label"), get_attr(attrs, "name"), get_attr(attrs, "placeholder")]:
        if not value:
            continue
        values.append(value)
        if "|" in value:
            values.extend(part.strip() for part in value.split("|") if part.strip())
        values.append(split_field_text(value))

    expanded = []
    for value in values:
        clean = re.sub(r"\s+", " ", value).strip(" *")
        if not clean:
            continue
        expanded.append(clean)
        words = split_field_text(clean).split()
        stripped = [w for w in words if w.lower() not in {"id", "code", "number", "date", "name"}]
        if stripped and stripped != words:
            expanded.append(" ".join(stripped))

    out = []
    seen = set()
    for value in expanded:
        key = value.lower()
        if key in seen:
            continue
        if re.fullmatch(r"[A-Z]{2,6}[- ][X0-9]+", value):
            continue
        if key in GENERIC_FIELD_WORDS:
            continue
        seen.add(key)
        out.append(value)
    return sorted(out, key=len, reverse=True)


def placeholder_regex(placeholder: str):
    placeholder = (placeholder or "").strip()
    if not placeholder:
        return None
    if placeholder.upper() == "YYYY-MM-DD":
        return ISO_DATE_RE
    match = re.fullmatch(r"([A-Z]{2,6})[- ]([X0]+)", placeholder, re.I)
    if match:
        prefix = match.group(1).upper()
        length = len(match.group(2))
        return re.compile(rf"\b{re.escape(prefix)}-[A-Z0-9]{{{length}}}\b")
    if "email" in placeholder.lower():
        return EMAIL_RE
    if placeholder.lower() in {"zip", "zipcode", "zip code"}:
        return re.compile(r"\b\d{5}(?:-\d{4})?\b")
    return None


def normalize_for_option_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def clean_extracted_value(value: str, cand: dict | None = None) -> str:
    value = (value or "").strip()
    value = re.sub(r"^Task:\s*", "", value, flags=re.I).strip()
    value = re.sub(r"\.\s+The\b.*$", "", value).strip()
    value = VALUE_TAIL_RE.sub("", value).strip()
    value = value.strip(" ,.;:")
    value = re.sub(r"\s+", " ", value)
    if not value:
        return ""

    if cand is not None:
        blob = field_blob(cand)
        if "email" in blob:
            match = EMAIL_RE.search(value)
            return match.group(0) if match else value
        if any(key in blob for key in ["date", "yyyy mm dd", "yyyy-mm-dd", "expiration", "due"]):
            match = ISO_DATE_RE.search(value)
            return match.group(0) if match else value
        if any(key in blob for key in ["amount", "quantity", "price", "min", "max", "pledge"]):
            money = MONEY_RE.search(value)
            if money:
                return money.group(1)
            match = DECIMAL_RE.search(value)
            return match.group(0) if match else value
        if any(
            key in blob
            for key in [
                " id",
                "_id",
                " code",
                "serial",
                "object",
                "application",
                "grant",
                "episode",
                "item code",
                "course code",
                "mission",
                "asset",
                "shipment",
                "robot",
                "sample",
                "pass id",
                "account id",
                "confirmation",
            ]
        ):
            match = CODE_RE.search(value)
            return match.group(0) if match else value
        if any(key in blob for key in ["airport", "origin", "destination"]) and re.fullmatch(r"[A-Z]{3}", value):
            return value.lower()

    value = re.sub(r"\s+\([A-Z]{3}\)$", "", value)
    return value.strip()


def extract_by_field_label(task: str, cand: dict) -> str:
    for label in candidate_field_strings(cand):
        label_re = re.escape(label)
        patterns = [
            rf"(?i)\b{label_re}\b\s*(?:is|as|to|for|:|=)?\s*([^,;\n]+)",
            rf"(?i)\b{label_re}\b\s+([A-Z0-9][^,;\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task or "")
            if not match:
                continue
            value = clean_extracted_value(match.group(1), cand)
            if value and value.lower() != label.lower():
                return value
    return ""


def extract_location_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    task = task or ""

    def capture(pattern: str) -> str:
        match = re.search(pattern, task, re.I)
        if not match:
            return ""
        value = match.group(1).strip(" ,.;:")
        value = re.sub(r"\s+\([A-Z]{3}\)$", "", value)
        if re.fullmatch(r"[A-Z]{3}", value):
            return value.lower()
        return value

    if any(key in blob for key in ["origin", "from", "departing from", "flying from"]):
        return capture(r"\bfrom\s+(.+?)(?=\s+to\b|\s+leaving\b|\s+depart|\s+search|,|\.)") or capture(
            r"\bdeparting from\s+(.+?)(?=\s+to\b|,|\.)"
        )
    if any(key in blob for key in ["destination", "flying to"]):
        return capture(r"\bto\s+(.+?)(?=\s+(?:search|leaving|depart|from|on|with)\b|,|\.)")
    if any(key in blob for key in ["pickup", "pick up", "pu search", "pusearch", "near", "town", "zipcode", "location", "city"]):
        return capture(r"\bin\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)(?=\s+from\b|\s+on\b|,|\.)") or capture(
            r"\bnear\s+(.+?)(?=\s+(?:from|on|with)\b|,|\.)"
        )
    return ""


def extract_name_part(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    task = task or ""
    full_name = ""
    patterns = [
        r"\bname\s+is\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)",
        r"\bfor\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)",
        r"\bpassenger\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)",
        r"\bassign\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, task)
        if match:
            full_name = match.group(1).strip()
            break
    if not full_name:
        return ""
    parts = full_name.split()
    if "first name" in blob and parts:
        return parts[0]
    if "last name" in blob and len(parts) >= 2:
        return parts[-1]
    if any(key in blob for key in ["technician", "assignee", "reviewer", "advisor", "pilot", "owner", "manager", "caller", "employee"]):
        return full_name
    return ""


def extract_search_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    task = task or ""
    if not any(key in blob for key in ["search", "query", "keyword", "field keywords"]):
        return ""
    patterns = [
        r"\bbuy\s+(?:a|an|the)?\s*(.+?)(?=\s+(?:in the color|in color|and have|with|from|on|at|near)\b|,|\.)",
        r"\bbrowse\s+(.+?)(?=\s+(?:that|which|available|with|from|on|in|at|near|rated|is)\b|,|\.)",
        r"\bplay\s+(?:a|an|the)?\s*(.+?)(?=\s+(?:trailer|video|movie|show)\b|,|\.)",
        r"\bsearch for\s+(.+?)(?=\s+(?:with|from|on|in|at|near)\b|,|\.)",
        r"\bfind\s+(.+?)(?=\s+(?:with|from|on|in|at|near)\b|,|\.)",
        r"\bfind\s+.+?\bof\s+(.+?)(?=\s+(?:and|with|from|on|in|at|near)\b|,|\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, task, re.I)
        if match:
            value = clean_extracted_value(match.group(1), cand)
            value = re.sub(r"\bsite_[a-z0-9]+\b", "", value, flags=re.I).strip()
            if value.lower() == "star wars":
                return "Star Wars"
            words = value.split()
            if words and words[-1].lower().endswith("s") and words[-1].lower() != "wars" and len(words[-1]) > 3:
                words[-1] = words[-1][:-1]
                value = " ".join(words)
            if value.islower():
                return value
            return value.title() if value else ""
    return ""


def extract_issue_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    if not any(key in blob for key in ["issue", "summary", "describe issue", "condition", "note"]):
        return ""
    match = re.search(r"\babout\s+(.+?)(?=,\s*(?:assign|set|schedule)|\s+assign\b|\s+set\b|,|\.)", task or "", re.I)
    return clean_extracted_value(match.group(1), cand).lower() if match else ""


def extract_address_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    cand_has_no_field_text = not (cand.get("text") or cand.get("attrs"))
    if "address" not in blob and not cand_has_no_field_text:
        return ""
    match = re.search(
        r"\baddress\s+(?:is\s+)?(.+?)(?=\.\s+The\s+(?:email|emial)\b|\s+(?:email|emial)\s+address\b| when asked\b|$)",
        task or "",
        re.I,
    )
    if not match:
        return ""
    value = match.group(1).strip(" ,.;:")
    value = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", value)
    return re.sub(r"\s+", " ", value)


def extract_quantity_context_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    task = task or ""
    if any(key in blob for key in ["attendee", "guest", "people", "passenger", "adult", "size", "room"]):
        match = re.search(r"\b(\d+)\s+(?:guests?|people|adults?|attendees?|passengers?)\b", task, re.I)
        if match:
            return match.group(1)
    if "mile" in blob or not (cand.get("text") or cand.get("attrs")):
        match = re.search(r"\b(?:approx(?:imately)?\s*)?(\d+)\s+miles?\b", task, re.I)
        if match:
            return match.group(1)
    if "zip" in blob or "postal" in blob or "store" in blob:
        match = re.search(r"\bzip(?: code)?\s+(\d{5}(?:-\d{4})?)\b", task, re.I) or re.search(r"\bfrom\s+(\d{5})\b", task, re.I)
        if match:
            return match.group(1)
    return ""


def extract_color_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    if not any(key in blob for key in ["color", "colour", "exterior"]):
        return ""
    colors = "black|white|red|blue|green|silver|gray|grey|yellow|orange|purple|brown|gold|magenta"
    match = re.search(rf"\b({colors})\s+(?:exterior|interior|color|colour)\b", task or "", re.I)
    if not match:
        match = re.search(rf"\bcolor\s+({colors})\b", task or "", re.I)
    return match.group(1).lower() if match else ""


def extract_org_value(task: str, cand: dict) -> str:
    blob = field_blob(cand)
    if not any(key in blob for key in ["org", "organization", "employer"]):
        return ""
    patterns = [
        r"\borganized by\s+([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*)",
        r"\bEmployer'?s name is\s+([^,.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, task or "")
        if match:
            return clean_extracted_value(match.group(1), cand)
    return ""


def fallback_type_value(task: str, cand: dict) -> str:
    for extractor in [
        extract_issue_value,
        extract_address_value,
        extract_quantity_context_value,
        extract_color_value,
        extract_org_value,
        extract_location_value,
        extract_name_part,
        extract_search_value,
    ]:
        value = extractor(task, cand)
        if value:
            return value

    text_blob = field_blob(cand)
    if "zip" in text_blob or "city state or zip" in text_blob:
        match = re.search(r"\b(?:from|zip(?: code)?|within)\s+(\d{5}(?:-\d{4})?)\b", task or "", re.I)
        if match:
            return match.group(1)

    for regex in [EMAIL_RE, ISO_DATE_RE, CODE_RE]:
        match = regex.search(task or "")
        if match:
            return clean_extracted_value(match.group(0), cand)

    quoted = re.search(r'"([^"]{1,80})"', task or "")
    if quoted:
        return clean_extracted_value(quoted.group(1), cand)

    money = MONEY_RE.search(task or "")
    if money:
        return money.group(1)

    number = DECIMAL_RE.search(task or "")
    if number:
        return number.group(0)

    words = [w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", task or "") if w.lower() not in STOPWORDS and not w.lower().startswith("site_")]
    return " ".join(words[:3]) if words else "value"


def fallback_select_value(task: str, cand: dict) -> str:
    options = parse_options(cand.get("attrs") or "")
    if options:
        return options[0]

    blob = field_blob(cand)
    task = task or ""
    if any(key in blob for key in ["lang", "language"]):
        match = re.search(r"\bin\s+([A-Z][A-Za-z]+)\s+with\s+ISBN\b", task)
        if match:
            return match.group(1)
    if any(key in blob for key in ["list create type", "list-create-type"]):
        if re.search(r"\b(?:director|actor|person|people|author|artist)\b", task, re.I):
            return "People"
    if "wish" in task.lower() and "list" in task.lower():
        return "Wish List"
    if any(key in blob for key in ["size", "guest"]):
        match = re.search(r"\bfor\s+(\d+)\s+people\b", task, re.I)
        if match:
            return f"{match.group(1)} guests"
    if any(key in blob for key in ["passenger", "adult"]):
        match = re.search(r"\bfor\s+(\d+)\s+adults?\b", task, re.I)
        if match:
            return match.group(1)
    if "type=number" in (cand.get("attrs") or "").lower():
        time = re.search(r"\b\d{1,2}:(\d{2})\b", task)
        if time:
            return str(int(time.group(1)))
        number = DECIMAL_RE.search(task)
        if number:
            return number.group(0)
    if "start time" in blob or "time" in blob:
        match = re.search(r"\b(?:after|at)\s+(\d{1,2})(?::?(\d{2}))?\s*(am|pm)\b", task, re.I)
        if match:
            minute = match.group(2) or "00"
            return f"{int(match.group(1))} {minute} {match.group(3).upper()}"
    if "season" in task.lower():
        match = re.search(r"\b(\d{4}-\d{2}\s+Regular Season)\b", task, re.I)
        if match:
            return match.group(1).title()

    text = (cand.get("text") or "").strip()
    if text and not re.fullmatch(r"[0-9a-f:.-]{12,}", text, re.I):
        return clean_extracted_value(text, cand)

    quoted = re.search(r'"([^"]{1,80})"', task)
    if quoted:
        return clean_extracted_value(quoted.group(1), cand)
    return "value"


def extract_value_type(task: str, cand: dict) -> str:
    task = task if isinstance(task, str) else ""
    attrs = cand.get("attrs") or ""
    blob = field_blob(cand)

    placeholder = get_attr(attrs, "placeholder")
    placeholder_re = placeholder_regex(placeholder)
    if placeholder_re is not None:
        match = placeholder_re.search(task)
        if match:
            return clean_extracted_value(match.group(0), cand)

    value = extract_by_field_label(task, cand)
    if value:
        return value

    if "email" in blob:
        match = EMAIL_RE.search(task)
        if match:
            return match.group(0)
    if any(key in blob for key in ["date", "yyyy mm dd", "yyyy-mm-dd", "expiration", "due"]):
        match = ISO_DATE_RE.search(task)
        if match:
            return match.group(0)

    value = extract_location_value(task, cand)
    if value:
        return value

    value = extract_name_part(task, cand)
    if value:
        return value

    value = extract_search_value(task, cand)
    if value:
        return value

    if any(
        key in blob
        for key in [
            " id",
            "_id",
            " code",
            "serial",
            "object",
            "application",
            "grant",
            "episode",
            "item code",
            "course code",
            "mission",
            "asset",
            "shipment",
            "robot",
            "sample",
            "pass id",
            "account id",
            "confirmation",
        ]
    ):
        match = CODE_RE.search(task)
        if match:
            return match.group(0)

    if any(key in blob for key in ["amount", "quantity", "price", "min", "max", "pledge"]):
        money = MONEY_RE.search(task)
        if money:
            return money.group(1)
        match = DECIMAL_RE.search(task)
        if match:
            return match.group(0)

    return fallback_type_value(task, cand)


def extract_value_select(task: str, cand: dict) -> str:
    options = parse_options(cand.get("attrs") or "")
    if not options:
        return fallback_select_value(task, cand)
    task_norm = f" {normalize_for_option_match(task)} "
    matches = []
    for option in options:
        option_norm = normalize_for_option_match(option)
        if option_norm and f" {option_norm} " in task_norm:
            matches.append(option)
    if matches:
        return max(matches, key=len)

    for option in options:
        tokens = normalize_for_option_match(option).split()
        if tokens and all(f" {token} " in task_norm for token in tokens):
            matches.append(option)
    return max(matches, key=len) if matches else fallback_select_value(task, cand)


def predict_value(task: str, cand: dict, op: str) -> str:
    if op == "CLICK":
        return ""
    if op == "SELECT":
        return extract_value_select(task, cand)
    if op == "TYPE":
        return extract_value_type(task, cand)
    return ""


def normalize_value(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def predict_from_scores(scores, group_sizes, meta_list):
    rows = []
    cursor = 0
    for size, meta in zip(group_sizes, meta_list):
        group_scores = scores[cursor : cursor + size]
        cursor += size
        best_pos = int(np.argmax(group_scores))
        pred_cand = meta["candidates"][best_pos]
        pred_id = pred_cand.get("candidate_id") or ""
        raw_cand = meta["raw_by_id"].get(pred_id, pred_cand)
        op = predict_op(raw_cand)
        value = predict_value(meta.get("task_raw") or meta.get("task"), raw_cand, op)

        true_pos = -1
        if meta.get("target_id") in meta.get("candidate_ids", []):
            true_pos = meta["candidate_ids"].index(meta["target_id"])
        order = np.argsort(-group_scores)
        true_rank = int(np.where(order == true_pos)[0][0]) + 1 if true_pos >= 0 else 999

        rows.append(
            {
                "id": meta["id"],
                "target_id": pred_id,
                "op": op,
                "value": value,
                "true_target_id": meta.get("target_id", ""),
                "true_op": meta.get("true_op", ""),
                "true_value": normalize_value(meta.get("true_value", "")),
                "target_rank": true_rank,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(pred: pd.DataFrame) -> dict[str, float]:
    pred = pred.copy()
    pred["target_match"] = pred["target_id"] == pred["true_target_id"]
    pred["op_match"] = pred["op"] == pred["true_op"]
    pred["value_match"] = pred["value"].map(normalize_value) == pred["true_value"].map(normalize_value)
    pred["all_match"] = pred["target_match"] & pred["op_match"] & pred["value_match"]
    metrics = {
        "target_id_acc": float(pred["target_match"].mean()),
        "target_top3_acc": float((pred["target_rank"] <= 3).mean()),
        "target_top5_acc": float((pred["target_rank"] <= 5).mean()),
        "op_acc": float(pred["op_match"].mean()),
        "value_acc": float(pred["value_match"].mean()),
        "all_match_acc": float(pred["all_match"].mean()),
    }
    correct_target = pred["target_match"]
    if correct_target.any():
        metrics["op_acc_given_target"] = float(pred.loc[correct_target, "op_match"].mean())
        metrics["value_acc_given_target"] = float(pred.loc[correct_target, "value_match"].mean())
    return metrics


def prepare_train_df() -> pd.DataFrame:
    train = load_csv(TRAIN_PATH)
    raw_cols = ["id", "task", "history", "cleaned_html", "candidate_elements"]
    raw_train = load_csv(RAW_TRAIN_PATH)[raw_cols].rename(
        columns={
            "task": "task_raw",
            "history": "history_raw",
            "cleaned_html": "cleaned_html_raw",
            "candidate_elements": "candidate_elements_raw",
        }
    )
    train = train.merge(raw_train, on="id", how="left")
    train["candidate_list"] = train["candidate_elements"].map(safe_json_loads)
    train["raw_candidate_list"] = train["candidate_elements_raw"].map(safe_json_loads)
    return train


def prepare_test_df() -> pd.DataFrame:
    test = load_csv(TEST_PATH)
    for col in ["op", "target_id", "value"]:
        if col not in test.columns:
            test[col] = ""
    raw_cols = ["id", "task", "history", "cleaned_html", "candidate_elements"]
    raw_test = load_csv(RAW_TEST_PATH)[raw_cols].rename(
        columns={
            "task": "task_raw",
            "history": "history_raw",
            "cleaned_html": "cleaned_html_raw",
            "candidate_elements": "candidate_elements_raw",
        }
    )
    test = test.merge(raw_test, on="id", how="left")
    test["candidate_list"] = test["candidate_elements"].map(safe_json_loads)
    test["raw_candidate_list"] = test["candidate_elements_raw"].map(safe_json_loads)
    return test


def fit_ranker(X_trn, y_trn, g_trn, X_val=None, y_val=None, g_val=None, n_estimators=2500):
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        label_gain=[0, 1],
        random_state=RNG_SEED,
        verbose=-1,
    )
    fit_kwargs = {
        "X": X_trn,
        "y": y_trn,
        "group": g_trn,
        "categorical_feature": [name for name in CATEGORICAL if name in X_trn.columns],
    }
    callbacks = []
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_group"] = [g_val]
        fit_kwargs["eval_at"] = [1, 3, 5]
        callbacks = [lgb.early_stopping(100), lgb.log_evaluation(100)]
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    ranker.fit(**fit_kwargs)
    return ranker


def save_submission(test: pd.DataFrame, scores, group_sizes, meta_list, out_path: Path) -> pd.DataFrame:
    pred = predict_from_scores(scores, group_sizes, meta_list)
    submission = test[["id"]].merge(pred[["id", "op", "target_id", "value"]], on="id", how="left")
    submission["op"] = submission["op"].fillna("CLICK")
    submission["target_id"] = submission["target_id"].fillna("")
    submission.loc[submission["op"] == "CLICK", "value"] = ""
    submission["value"] = submission["value"].fillna("")
    submission[["id", "op", "target_id", "value"]].to_csv(out_path, index=False, lineterminator="\n")
    return submission


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TRAIN_PATH: {TRAIN_PATH}")
    print(f"TEST_PATH: {TEST_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    train = prepare_train_df()
    print("train shape:", train.shape)

    idx = np.arange(len(train))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=RNG_SEED,
        stratify=train["op"] if train["op"].nunique() > 1 else None,
    )
    trn_df = train.iloc[train_idx].reset_index(drop=True)
    val_df = train.iloc[val_idx].reset_index(drop=True)

    idf = compute_idf((trn_df["task"].fillna("") + " " + trn_df["history"].fillna("")).tolist())
    site_freq = trn_df["site_token"].value_counts(normalize=True).to_dict()
    prior_stats = compute_prior_stats(trn_df)

    print("[features] train/valid")
    X_trn, y_trn, g_trn, _ = build_rank_dataset(trn_df, idf, site_freq, prior_stats)
    X_val, y_val, g_val, meta_val = build_rank_dataset(val_df, idf, site_freq, prior_stats)
    X_trn_model = X_trn[MODEL_FEAT_NAMES]
    X_val_model = X_val[MODEL_FEAT_NAMES]
    print("X_trn:", X_trn.shape, "X_val:", X_val.shape)
    print("model features:", len(MODEL_FEAT_NAMES))

    print("[fit] holdout model")
    ranker = fit_ranker(X_trn_model, y_trn, g_trn, X_val_model, y_val, g_val)
    val_scores = ranker.predict(X_val_model, num_iteration=ranker.best_iteration_)
    pred_val = predict_from_scores(val_scores, g_val, meta_val)
    metrics = evaluate_predictions(pred_val)

    print("[metrics]")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    pred_val.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    value_errors = pred_val[
        (pred_val["true_op"] != "CLICK")
        & (pred_val["target_id"] == pred_val["true_target_id"])
        & (pred_val["value"].map(normalize_value) != pred_val["true_value"].map(normalize_value))
    ].copy()
    value_errors.head(200).to_csv(OUTPUT_DIR / "value_error_samples.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEAT_NAMES,
            "gain": ranker.booster_.feature_importance(importance_type="gain"),
            "split": ranker.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    importance.to_csv(SCRIPT_DIR / "feature_importance.csv", index=False)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(SCRIPT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[refit] full train")
    idf_all = compute_idf((train["task"].fillna("") + " " + train["history"].fillna("")).tolist())
    site_freq_all = train["site_token"].value_counts(normalize=True).to_dict()
    prior_stats_all = compute_prior_stats(train)
    X_all, y_all, g_all, _ = build_rank_dataset(train, idf_all, site_freq_all, prior_stats_all)
    X_all_model = X_all[MODEL_FEAT_NAMES]
    final_iters = int((ranker.best_iteration_ or 500) * 1.15)
    print("best_iteration:", ranker.best_iteration_, "final_iters:", final_iters)
    final_ranker = fit_ranker(X_all_model, y_all, g_all, n_estimators=final_iters)

    bundle = {
        "model": final_ranker,
        "feature_names": MODEL_FEAT_NAMES,
        "all_feature_names": FEAT_NAMES,
        "categorical": [name for name in CATEGORICAL if name in MODEL_FEAT_NAMES],
        "idf": idf_all,
        "site_freq": site_freq_all,
        "prior_stats": prior_stats_all,
        "tag_vocab": TAG_VOCAB,
        "input_type_vocab": INPUT_TYPE_VOCAB,
        "op_vocab": OP_VOCAB,
        "metrics": metrics,
    }
    joblib.dump(bundle, OUTPUT_DIR / "lgbm_ranker_bundle.joblib")

    if TEST_PATH.exists() and RAW_TEST_PATH.exists():
        print("[predict] test")
        test = prepare_test_df()
        X_test, _, g_test, meta_test = build_rank_dataset(test, idf_all, site_freq_all, prior_stats_all)
        X_test_model = X_test[MODEL_FEAT_NAMES]
        test_scores = final_ranker.predict(X_test_model) if len(X_test_model) else np.zeros(0)
        submission = save_submission(test, test_scores, g_test, meta_test, SCRIPT_DIR / "submission_lgbm_ranker_v2.csv")
        submission.to_csv(OUTPUT_DIR / "submission_lgbm_ranker_v2.csv", index=False, lineterminator="\n")
        print("submission shape:", submission.shape)
        print(submission["op"].value_counts())

    print("saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
