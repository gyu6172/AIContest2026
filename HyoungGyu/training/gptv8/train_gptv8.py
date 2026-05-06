from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEED = 20260506
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9._:/+,-]*[A-Za-z0-9])?")
CAMEL_RE = re.compile(r"([a-z])([A-Z])")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
CODE_RE = re.compile(r"\b[A-Z]{1,8}-(?=[A-Z0-9]*\d)[A-Z0-9]{2,24}\b", re.I)
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
ROOM_RE = re.compile(r"\b[A-Z]-\d{3,4}\b")
CAP_NAME_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")

STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "task",
    "step",
    "click",
    "type",
    "select",
    "selected",
    "enter",
    "choose",
    "set",
    "find",
    "show",
}
CLICK_INPUT_TYPES = {"button", "checkbox", "image", "radio", "range", "reset", "submit"}

COMPLETED_RE = re.compile(r"Completed:\s*([^<\n]+)", re.I)
STEP_RE = re.compile(r"step\s+(\d+)\s+of\s+(\d+)", re.I)
HIST_STEP_RE = re.compile(
    r"Step\s+\d+:\s*\[([^\]]*)\]\s*(.*?)\s*->\s*(CLICK|TYPE|SELECT)(?::\s*([^\n]*))?",
    re.I,
)
PANEL_RE = re.compile(
    r'<section[^>]*aria-label="current workflow panel"[^>]*>(.*?)</section>',
    re.I | re.S,
)
NAME_HTML_RE = re.compile(r'\bname="([^"]+)"', re.I)
H1_RE = re.compile(r"<h1[^>]*>([^<]+)</h1>", re.I)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "train.csv").exists() and (
            candidate / "data" / "test.csv"
        ).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find project root with data/train.csv and data/test.csv"
    )


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = find_project_root(SCRIPT_DIR)
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "HyoungGyu" / "training" / "gptv8" / "submission.csv"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, encoding="utf-8", encoding_errors="replace", low_memory=False
    )


def norm_key(value: Any) -> str:
    text = as_text(value).replace("_", " ")
    text = CAMEL_RE.sub(r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def norm_text(value: Any) -> str:
    text = as_text(value).replace("_", " ")
    text = CAMEL_RE.sub(r"\1 \2", text)
    text = re.sub(r"[^A-Za-z0-9._:/+,-]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    out: list[str] = []
    for match in TOKEN_RE.findall(as_text(value)):
        for token in norm_key(match).split():
            if len(token) > 1 and token not in STOPWORDS:
                out.append(token)
    return out


def parse_attrs(value: Any) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in as_text(value).split(" | "):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        if key:
            attrs[key] = val.strip()
    return attrs


def load_candidates(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(as_text(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def parse_options(attrs: dict[str, str]) -> list[str]:
    return [
        part.strip() for part in attrs.get("options", "").split(" / ") if part.strip()
    ]


def clean_value(value: Any) -> str:
    text = as_text(value).strip().strip(' ,.;:"')
    return re.sub(r"\s+", " ", text)


def candidate_labels(
    cand: dict[str, Any],
    attrs: dict[str, str] | None = None,
    include_options: bool = False,
) -> list[str]:
    attrs = attrs if attrs is not None else parse_attrs(cand.get("attrs", ""))
    vals: list[str] = []
    for value in [
        cand.get("text", ""),
        attrs.get("label", ""),
        attrs.get("placeholder", ""),
        attrs.get("name", ""),
        attrs.get("text", ""),
        attrs.get("alt", ""),
        attrs.get("aria-label", ""),
        attrs.get("title", ""),
        attrs.get("value", ""),
    ]:
        if clean_value(value):
            vals.append(clean_value(value))
            key = norm_key(value)
            if key:
                vals.append(key)

    if include_options:
        vals.extend(parse_options(attrs))

    expanded: list[str] = []
    for value in vals:
        parts = norm_key(value).split()
        if len(parts) > 1:
            expanded.append(" ".join(parts))
            if parts[-1] in {
                "id",
                "number",
                "date",
                "name",
                "code",
                "amount",
                "status",
                "level",
                "flag",
                "mode",
                "type",
                "priority",
            }:
                expanded.append(" ".join(parts[:-1]))
            expanded.append(parts[0])
    vals.extend(expanded)

    seen: set[str] = set()
    out: list[str] = []
    for value in vals:
        normalized = norm_text(value)
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        out.append(value)
    out.sort(key=lambda item: len(norm_text(item)), reverse=True)
    return out


def phrase_pos(blob_norm: str, phrase: Any) -> int:
    normalized = norm_text(phrase)
    if len(normalized) < 2:
        return -1
    match = re.search(
        r"(?<![a-z0-9])" + re.escape(normalized) + r"(?![a-z0-9])", blob_norm
    )
    return -1 if match is None else match.start()


def first_phrase_pos(blob_norm: str, labels: list[str]) -> int:
    positions = [phrase_pos(blob_norm, label) for label in labels]
    positions = [pos for pos in positions if pos >= 0]
    return min(positions) if positions else -1


def row_context(row: pd.Series) -> dict[str, Any]:
    task = as_text(row.get("task"))
    history = as_text(row.get("history"))
    html = as_text(row.get("cleaned_html"))
    task_tokens = set(tokenize(task))
    history_tokens = set(tokenize(history))

    completed: set[str] = set()
    for match in COMPLETED_RE.finditer(html):
        for part in match.group(1).split(","):
            item = norm_key(part)
            if item:
                completed.add(item)

    history_labels: list[str] = []
    history_values: list[str] = []
    for match in HIST_STEP_RE.finditer(history):
        label = norm_key(match.group(2))
        if label:
            history_labels.append(label)
        if match.group(4):
            history_values.append(norm_key(match.group(4)))

    step_match = STEP_RE.search(html)
    cur_step = int(step_match.group(1)) if step_match else 0
    total_step = int(step_match.group(2)) if step_match else 0

    panel_names: set[str] = set()
    panel_match = PANEL_RE.search(html)
    if panel_match:
        panel_names = {
            norm_key(name) for name in NAME_HTML_RE.findall(panel_match.group(1))
        }

    h1_tokens: set[str] = set()
    for match in H1_RE.finditer(html):
        h1_tokens |= set(tokenize(match.group(1)))

    return {
        "task": task,
        "history": history,
        "html": html,
        "task_norm": norm_text(task),
        "history_norm": norm_text(history),
        "T": task_tokens,
        "H": history_tokens,
        "R": task_tokens - history_tokens,
        "completed": completed,
        "history_labels": history_labels,
        "history_values": history_values,
        "n_history_steps": len(history_labels),
        "cur_step": cur_step,
        "total_step": total_step,
        "panel_names": panel_names,
        "h1_tokens": h1_tokens,
        "is_workflow": int(
            "workflow-context" in html or "current workflow panel" in html
        ),
        "html_len_log": math.log1p(len(html)),
    }


def candidate_token_set(cand: dict[str, Any], attrs: dict[str, str]) -> set[str]:
    parts = [cand.get("tag", ""), cand.get("text", ""), cand.get("attrs", "")]
    for key, value in attrs.items():
        parts.extend([key, value])
        if key == "name":
            parts.append(norm_key(value))
    return set(tokenize(" ".join(as_text(part) for part in parts)))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def dense_rank(values: list[float], reverse: bool = True) -> list[int]:
    arr = np.array(values, dtype=float)
    order = np.argsort(-arr if reverse else arr, kind="stable")
    ranks = np.empty(len(arr), dtype=int)
    ranks[order] = np.arange(len(arr))
    return ranks.tolist()


def compute_idf(train_df: pd.DataFrame) -> dict[str, float]:
    freq: Counter[str] = Counter()
    total = 0
    for _, row in train_df.iterrows():
        total += 1
        for token in row_context(row)["T"]:
            freq[token] += 1
    return {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in freq.items()
    }


def build_candidate_features(
    df: pd.DataFrame, *, include_label: bool, idf: dict[str, float]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        ctx = row_context(row)
        candidates = load_candidates(row.get("candidate_elements"))
        target_id = as_text(row.get("target_id")) if include_label else ""
        row_records: list[dict[str, Any]] = []

        for pos, cand in enumerate(candidates):
            attrs = parse_attrs(cand.get("attrs", ""))
            labels = candidate_labels(cand, attrs)
            options = parse_options(attrs)
            cand_tokens = candidate_token_set(cand, attrs)

            inter_t = len(ctx["T"] & cand_tokens)
            inter_h = len(ctx["H"] & cand_tokens)
            inter_r = len(ctx["R"] & cand_tokens)
            idf_t = sum(idf.get(token, 1.0) for token in (ctx["T"] & cand_tokens))
            idf_r = sum(idf.get(token, 1.0) for token in (ctx["R"] & cand_tokens))

            label_task_pos = first_phrase_pos(ctx["task_norm"], labels)
            label_history_pos = first_phrase_pos(ctx["history_norm"], labels)
            text_key = norm_key(cand.get("text", ""))
            label_key = norm_key(attrs.get("label", "") or cand.get("text", ""))
            name_key = norm_key(attrs.get("name", ""))
            placeholder_key = norm_key(attrs.get("placeholder", ""))

            option_positions = [
                phrase_pos(ctx["task_norm"], option) for option in options
            ]
            option_hits = sum(pos >= 0 for pos in option_positions)
            positive_option_positions = [pos for pos in option_positions if pos >= 0]
            option_best_pos = (
                min(positive_option_positions) if positive_option_positions else -1
            )

            in_completed = int(
                any(
                    value
                    and (value in ctx["completed"] or value in ctx["history_labels"])
                    for value in [text_key, label_key, name_key]
                )
            )
            in_history_value = int(
                any(
                    value and value in ctx["history_values"]
                    for value in [text_key, label_key, name_key]
                )
            )
            name_in_panel = int(bool(name_key and name_key in ctx["panel_names"]))

            tag = norm_key(cand.get("tag", "")) or "<empty>"
            input_type = norm_key(attrs.get("type", "")) or "<none>"
            role = norm_key(attrs.get("role", "")) or "<none>"

            row_records.append(
                {
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "candidate_id": cand.get("candidate_id", ""),
                    "label": (
                        int(cand.get("candidate_id") == target_id)
                        if include_label
                        else 0
                    ),
                    "site": as_text(row.get("site_token")),
                    "tag": tag,
                    "type": input_type,
                    "role": role,
                    "name": name_key or "<none>",
                    "label_text": label_key or "<none>",
                    "text_key": text_key or "<none>",
                    "placeholder": placeholder_key or "<none>",
                    "is_workflow": ctx["is_workflow"],
                    "pos": pos,
                    "pos_norm": safe_div(pos, max(len(candidates) - 1, 1)),
                    "n_cands": len(candidates),
                    "n_task_tok": len(ctx["T"]),
                    "n_history_tok": len(ctx["H"]),
                    "n_remaining_tok": len(ctx["R"]),
                    "n_cand_tok": len(cand_tokens),
                    "inter_task": inter_t,
                    "inter_history": inter_h,
                    "inter_remaining": inter_r,
                    "cov_task": safe_div(inter_t, len(ctx["T"])),
                    "cov_history": safe_div(inter_h, len(ctx["H"])),
                    "cov_remaining": safe_div(inter_r, len(ctx["R"])),
                    "cand_cov_task": safe_div(inter_t, len(cand_tokens)),
                    "cand_cov_remaining": safe_div(inter_r, len(cand_tokens)),
                    "jac_task": safe_div(inter_t, len(ctx["T"] | cand_tokens)),
                    "jac_remaining": safe_div(inter_r, len(ctx["R"] | cand_tokens)),
                    "idf_task": idf_t,
                    "idf_remaining": idf_r,
                    "label_task_pos": label_task_pos if label_task_pos >= 0 else 99999,
                    "label_history_pos": (
                        label_history_pos if label_history_pos >= 0 else 99999
                    ),
                    "label_in_task": int(label_task_pos >= 0),
                    "label_in_history": int(label_history_pos >= 0),
                    "in_completed": in_completed,
                    "in_history_value": in_history_value,
                    "name_in_panel": name_in_panel,
                    "option_hits": option_hits,
                    "n_options": len(options),
                    "option_best_pos": (
                        option_best_pos if option_best_pos >= 0 else 99999
                    ),
                    "text_len": len(as_text(cand.get("text", ""))),
                    "attrs_len": len(as_text(cand.get("attrs", ""))),
                    "has_attrs": int(bool(as_text(cand.get("attrs", "")).strip())),
                    "cur_step": ctx["cur_step"],
                    "total_step": ctx["total_step"],
                    "step_remaining": max(ctx["total_step"] - ctx["cur_step"], 0),
                    "n_history_steps": ctx["n_history_steps"],
                    "html_len_log": ctx["html_len_log"],
                    "h1_overlap": len(ctx["h1_tokens"] & cand_tokens),
                    "is_clickish_input": int(
                        tag == "input" and input_type in CLICK_INPUT_TYPES
                    ),
                    "is_typeish": int(
                        tag in {"input", "textarea"}
                        and input_type not in CLICK_INPUT_TYPES
                    ),
                    "is_selectish": int(tag == "select"),
                    "is_buttonish": int(
                        tag
                        in {
                            "button",
                            "a",
                            "div",
                            "li",
                            "span",
                            "svg",
                            "img",
                            "label",
                            "td",
                        }
                    ),
                }
            )

        if row_records:
            rank_specs = [
                ("cov_remaining", True),
                ("cov_task", True),
                ("cand_cov_remaining", True),
                ("idf_remaining", True),
                ("label_task_pos", False),
                ("option_hits", True),
            ]
            for key, reverse in rank_specs:
                for rec, rank in zip(
                    row_records, dense_rank([r[key] for r in row_records], reverse)
                ):
                    rec[f"{key}_rank"] = rank

            remaining_positions = [
                (
                    rec["label_task_pos"]
                    if rec["label_in_task"] and not rec["in_completed"]
                    else 99999
                )
                for rec in row_records
            ]
            for rec, rank, remaining_pos in zip(
                row_records,
                dense_rank(remaining_positions, reverse=False),
                remaining_positions,
            ):
                rec["remaining_task_pos_rank"] = rank
                rec["remaining_task_pos"] = remaining_pos

        records.extend(row_records)

    return pd.DataFrame(records)


def group_sizes(feature_df: pd.DataFrame) -> list[int]:
    return feature_df.groupby("row_idx", sort=False).size().tolist()


def cast_categories(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame | None,
    test_x: pd.DataFrame,
    cat_cols: list[str],
) -> None:
    for col in cat_cols:
        values = [train_x[col], test_x[col]]
        if val_x is not None:
            values.append(val_x[col])
        dtype = pd.CategoricalDtype(
            categories=pd.Index(pd.concat(values).astype(str).unique())
        )
        train_x[col] = train_x[col].astype(str).astype(dtype)
        test_x[col] = test_x[col].astype(str).astype(dtype)
        if val_x is not None:
            val_x[col] = val_x[col].astype(str).astype(dtype)


def predict_targets(feature_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    pred = feature_df[["row_idx", "id", "candidate_id"]].copy()
    pred["score"] = scores
    best_idx = pred.groupby("row_idx", sort=False)["score"].idxmax()
    return pred.loc[best_idx].sort_values("row_idx").reset_index(drop=True)


def build_candidate_map(df: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in df.iterrows():
        out[row["id"]] = {
            cand.get("candidate_id"): cand
            for cand in load_candidates(row.get("candidate_elements"))
        }
    return out


def build_op_tables(train_df: pd.DataFrame):
    by_tag_type_role: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    by_tag_type: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_tag: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for _, row in train_df.iterrows():
        target = None
        for cand in load_candidates(row.get("candidate_elements")):
            if cand.get("candidate_id") == row.get("target_id"):
                target = cand
                break
        if target is None:
            continue

        attrs = parse_attrs(target.get("attrs", ""))
        tag = norm_key(target.get("tag", "")) or "<empty>"
        input_type = norm_key(attrs.get("type", "")) or "<none>"
        role = norm_key(attrs.get("role", "")) or "<none>"
        op = as_text(row.get("op"))
        by_tag_type_role[(tag, input_type, role)][op] += 1
        by_tag_type[(tag, input_type)][op] += 1
        by_tag[tag][op] += 1

    return by_tag_type_role, by_tag_type, by_tag


def majority(counter: Counter[str], min_count: int = 1) -> str | None:
    if not counter or sum(counter.values()) < min_count:
        return None
    return counter.most_common(1)[0][0]


def predict_op(cand: dict[str, Any], op_tables) -> str:
    by_tag_type_role, by_tag_type, by_tag = op_tables
    attrs = parse_attrs(cand.get("attrs", ""))
    tag = norm_key(cand.get("tag", "")) or "<empty>"
    input_type = norm_key(attrs.get("type", "")) or "<none>"
    role = norm_key(attrs.get("role", "")) or "<none>"

    for key, table in [
        ((tag, input_type, role), by_tag_type_role),
        ((tag, input_type), by_tag_type),
        (tag, by_tag),
    ]:
        op = majority(table.get(key, Counter()), min_count=2)
        if op:
            return op

    if tag == "select":
        return "SELECT"
    if tag in {"input", "textarea"} and input_type not in CLICK_INPUT_TYPES:
        return "TYPE"
    return "CLICK"


def option_match(task: str, options: list[str]) -> str:
    matches = []
    for option in options:
        if re.search(
            r"(?<![A-Za-z0-9])" + re.escape(option) + r"(?![A-Za-z0-9])", task, re.I
        ):
            matches.append((len(option), option))
    return max(matches)[1] if matches else ""


def extract_after_label(task: str, labels: list[str]) -> str:
    for label in labels:
        normalized = norm_key(label)
        if len(normalized) < 3:
            continue
        words = normalized.split()
        patterns = [r"\b" + r"\s+".join(map(re.escape, words)) + r"\b"]
        if len(words) > 1 and words[-1] in {"id", "number", "date", "name", "code"}:
            patterns.append(r"\b" + r"\s+".join(map(re.escape, words[:-1])) + r"\b")

        for pattern in patterns:
            match = re.search(
                pattern + r"\s*(?:is|as|to|at|of|=|:)?\s*([^,\.\n;]+)",
                task,
                re.I,
            )
            if not match:
                continue
            value = clean_value(match.group(1))
            value = re.sub(
                r"\s+and\s+(queue|submit|save|send|publish|create|open|click|approve|dispatch)\b.*$",
                "",
                value,
                flags=re.I,
            )
            value = clean_value(value)
            if value and norm_key(value) != normalized:
                return value
    return ""


def month_date(task: str) -> str:
    match = re.search(
        r"\b(?:on|from|leaving|returning|start(?:ing)?(?: on)?)\s+"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?|"
        r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b",
        task,
        re.I,
    )
    return clean_value(match.group(1)) if match else ""


def capitalized_name(task: str) -> str:
    match = re.search(
        r"\b(?:named|name:?|for|guest is named)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b", task
    )
    if match:
        return match.group(1)
    match = CAP_NAME_RE.search(task)
    return match.group(0) if match else ""


def location_after(task: str, preps: list[str]) -> str:
    prep = "|".join(map(re.escape, preps))
    patterns = [
        rf"\b(?:{prep})\s+([A-Z][A-Za-z .-]+?,\s*[A-Z]{{2}})\b",
        rf"\b(?:{prep})\s+([A-Z][A-Za-z .-]+?)(?:\s+(?:on|at|for|from|to|with|and|that|which|leaving|returning)\b|[,.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, task, re.I)
        if not match:
            continue
        value = clean_value(match.group(1))
        if value and value.lower() not in {"the", "a", "an"}:
            return value
    return ""


def query_from_task(task: str) -> str:
    patterns = [
        r"\bsearch(?: for)?\s+(.+?)(?:\s+(?:and|then|with|under|available|sorted|from|for|on|in)\b|[,.]|$)",
        r"\bshow reviews for\s+(.+?)(?:\s+and\b|[,.]|$)",
        r"\bcheck reviews and research information about\s+(.+?)(?:[,.]|$)",
        r"\bcheck prices of\s+(.+?)(?:\s+with\b|\s+and\b|[,.]|$)",
        r"\bfind\s+(.+?)(?:\s+(?:with|that|which|under|near|from|for|on|and|then|available|sorted|made|in)\b|[,.]|$)",
        r"\bbrowse\s+(.+?)(?:\s+(?:from|in|at|on|with|and|leaving)\b|[,.]|$)",
        r"\badd\s+(.+?)(?:\s+(?:to cart|with|from|and one|and|for|below|under)\b|[,.]|$)",
        r"\bfollow\s+(.+?)(?:\s+(?:from|in)\b|[,.]|$)",
    ]
    drop = {
        "the",
        "a",
        "an",
        "cheapest",
        "lowest",
        "highest",
        "best",
        "top",
        "first",
        "one",
        "two",
        "three",
        "different",
        "available",
        "used",
        "working",
    }
    for pattern in patterns:
        match = re.search(pattern, task, re.I)
        if not match:
            continue
        words = clean_value(match.group(1)).split()
        while words and words[0].lower() in drop:
            words = words[1:]
        value = clean_value(" ".join(words))
        if value and len(value) <= 80:
            return value
    return ""


def pipe_value(text: str) -> str:
    parts = [
        clean_value(part) for part in as_text(text).split("|") if clean_value(part)
    ]
    for part in reversed(parts):
        if not re.fullmatch(
            r"(search|input|all|null|select|type|country|price|format|availability|city|departure|destination|origin)",
            part,
            re.I,
        ):
            return part[:80]
    return ""


def type_fallback(task: str, cand: dict[str, Any], attrs: dict[str, str]) -> str:
    key = " ".join(
        [
            norm_key(cand.get("text", "")),
            norm_key(attrs.get("label", "")),
            norm_key(attrs.get("name", "")),
            norm_key(attrs.get("placeholder", "")),
        ]
    )

    if attrs.get("value") and norm_key(attrs["value"]) not in {
        "false",
        "true",
        "0",
        "1",
    }:
        return clean_value(attrs["value"])
    if "employee" in key or "applicant" in key:
        match = re.search(r"\bfor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b", task)
        if match:
            return match.group(1)
    if "technician" in key or "assignee" in key:
        match = re.search(r"\bassign\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b", task)
        if match:
            return match.group(1)
    if "issue" in key or "summary" in key:
        match = re.search(r"\babout\s+([^,.]+)", task, re.I)
        if match:
            return clean_value(match.group(1))
    if "room" in key:
        match = ROOM_RE.search(task)
        if match:
            return match.group(0)
    if "first name" in key or "firstname" in key:
        name = capitalized_name(task)
        return name.split()[0] if name else "Joe"
    if "last name" in key or "lastname" in key:
        name = capitalized_name(task)
        return name.split()[-1] if name else "Bloggs"
    if "email" in key:
        match = EMAIL_RE.search(task)
        if match:
            return match.group(0)
    if "phone" in key or "tel" in key:
        numbers = NUM_RE.findall(task)
        return max(numbers, key=len) if numbers else ""
    if "street2" in key:
        return ""
    if "street" in key or "address" in key:
        match = re.search(r"address is(?: in)?\s+([^,.]+)", task, re.I)
        return (
            clean_value(match.group(1))
            if match
            else location_after(task, ["near", "at", "in"])
        )
    if key.strip() == "city" or " city" in key:
        return location_after(task, ["in", "near", "at", "to"])
    if any(token in key for token in ["from", "origin", "departure", "flying from"]):
        return location_after(task, ["from", "leaving from"]) or pipe_value(
            cand.get("text", "")
        )
    if any(token in key for token in ["to", "destination", "drop", "return"]):
        return location_after(
            task, ["to", "drop off at", "return at", "return to"]
        ) or pipe_value(cand.get("text", ""))
    if any(
        token in key
        for token in ["where", "location", "pickup", "pick up", "parking", "near"]
    ):
        return (
            location_after(task, ["near", "at", "in", "on", "to"])
            or query_from_task(task)
            or pipe_value(cand.get("text", ""))
        )
    if "date" in key or "mm dd yyyy" in key or "birth" in key:
        match = ISO_DATE_RE.search(task)
        return match.group(0) if match else month_date(task)
    if any(
        token in key
        for token in [
            "maximum",
            "minimum",
            "value",
            "price",
            "budget",
            "amount",
            "down payment",
        ]
    ):
        dollars = re.findall(r"[$]\s*(\d+(?:\.\d+)?)", task)
        numbers = NUM_RE.findall(task)
        return dollars[-1] if dollars else (numbers[-1] if numbers else "")
    if any(
        token in key
        for token in [
            "attendees",
            "guest",
            "adult",
            "child",
            "age",
            "quantity",
            "qty",
            "count",
            "term",
            "month",
        ]
    ):
        numbers = NUM_RE.findall(task)
        return numbers[-1] if numbers else ""
    if any(token in key for token in ["search", "query", "keyword", "title", "q"]):
        return query_from_task(task) or pipe_value(cand.get("text", ""))
    if any(token in key for token in ["airport", "station"]):
        match = re.search(r"\b[A-Z]{3,4}\b", task)
        return (
            match.group(0)
            if match
            else location_after(task, ["from", "to", "at", "in"])
        )
    if any(
        token in key
        for token in [
            "id",
            "code",
            "confirmation",
            "part number",
            "order number",
            "plate",
        ]
    ):
        match = CODE_RE.search(task)
        if match:
            return match.group(0)
        match = NUM_RE.search(task)
        return match.group(0) if match else ""

    return (
        pipe_value(cand.get("text", ""))
        or query_from_task(task)
        or location_after(task, ["in", "near", "at", "to", "from"])
    )


def select_fallback(task: str, cand: dict[str, Any], attrs: dict[str, str]) -> str:
    key = " ".join(
        [
            norm_key(cand.get("text", "")),
            norm_key(attrs.get("label", "")),
            norm_key(attrs.get("name", "")),
        ]
    )
    options = parse_options(attrs)
    value = option_match(task, options)
    if value:
        return clean_value(value)
    if "make" in key:
        for brand in [
            "Toyota",
            "Audi",
            "BMW",
            "Ford",
            "Honda",
            "Hyundai",
            "Nissan",
            "Chevrolet",
            "Mercedes",
            "Lexus",
        ]:
            if re.search(r"\b" + brand + r"\b", task, re.I):
                return brand
    if "model" in key:
        match = re.search(
            r"\b(?:Toyota|Audi|BMW|Ford|Honda|Hyundai|Nissan|Chevrolet|Mercedes|Lexus)\s+([A-Z][A-Za-z0-9-]+)\b",
            task,
        )
        if match:
            return match.group(1)
    if "country" in key:
        if re.search(r"United States|New York", task, re.I):
            return "United States"
        if re.search(r"Auckland|New Zealand", task, re.I):
            return "New Zealand"
    if "format" in key and re.search(r"digital", task, re.I):
        return "Digital"
    if "availability" in key and re.search(r"in stock", task, re.I):
        return "In stock"
    if "condition" in key:
        if re.search(r"no defect|working", task, re.I):
            return "No defect"
        if re.search(r"used", task, re.I):
            return "Used"
    if "sort" in key or "price" in key:
        if re.search(r"cheapest|lowest|low-priced|low price", task, re.I):
            return "Lowest price"
        if re.search(r"highest|top|best", task, re.I):
            return "Highest rated"
    if "year" in key or "date" in key:
        years = re.findall(r"\b(?:19|20)\d{2}\b", task)
        return years[0] if years else month_date(task)
    if "ticket" in key:
        numbers = NUM_RE.findall(task)
        return numbers[0] if numbers else ""
    if any(
        token in key
        for token in [
            "hour",
            "minute",
            "guest",
            "adult",
            "child",
            "age",
            "quantity",
            "qty",
            "month",
            "term",
        ]
    ):
        numbers = NUM_RE.findall(task)
        return numbers[-1] if numbers else ""
    return (
        pipe_value(cand.get("text", ""))
        or query_from_task(task)
        or location_after(task, ["in", "near", "at", "to", "from"])
    )


def generic_nonempty_value(task: str, cand: dict[str, Any]) -> str:
    value = pipe_value(cand.get("text", ""))
    if value:
        return value
    for pattern in [
        r'"([^"]+)"',
        r"\b(?:for|from|to|in|near|at|about)\s+([^,.]+)",
        r"\b([A-Z]{2,5}-[A-Z0-9]{2,20})\b",
        r"\b(\d+(?:\.\d+)?)\b",
    ]:
        match = re.search(pattern, task, re.I)
        if match:
            value = clean_value(match.group(1))
            if value:
                return value[:80]
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9]+", task)
        if word.lower()
        not in {
            "task",
            "find",
            "show",
            "get",
            "the",
            "a",
            "an",
            "with",
            "and",
            "for",
            "to",
            "in",
            "on",
        }
    ]
    return " ".join(words[:3]) if words else "0"


def predict_value(task: str, cand: dict[str, Any], op: str) -> str:
    attrs = parse_attrs(cand.get("attrs", ""))
    if op == "CLICK":
        return ""
    if op == "SELECT":
        return clean_value(
            option_match(task, parse_options(attrs))
            or extract_after_label(task, candidate_labels(cand, attrs))
            or select_fallback(task, cand, attrs)
            or generic_nonempty_value(task, cand)
        )
    return clean_value(
        extract_after_label(task, candidate_labels(cand, attrs))
        or type_fallback(task, cand, attrs)
        or generic_nonempty_value(task, cand)
    )


def attach_op_value(
    target_pred: pd.DataFrame,
    source_df: pd.DataFrame,
    candidate_map: dict[str, dict[str, dict[str, Any]]],
    op_tables,
) -> pd.DataFrame:
    row_by_id = {row["id"]: row for _, row in source_df.iterrows()}
    ops: list[str] = []
    values: list[str] = []

    for _, pred in target_pred.iterrows():
        row_id = pred["id"]
        cand_id = pred["candidate_id"]
        cand = candidate_map.get(row_id, {}).get(cand_id, {})
        op = predict_op(cand, op_tables) if cand else "CLICK"
        value = (
            predict_value(as_text(row_by_id[row_id].get("task")), cand, op)
            if cand
            else ""
        )
        if op == "CLICK":
            value = ""
        ops.append(op)
        values.append(value)

    out = (
        target_pred[["id", "candidate_id"]]
        .rename(columns={"candidate_id": "target_id"})
        .copy()
    )
    out.insert(1, "op", ops)
    out["value"] = values
    return out[["id", "op", "target_id", "value"]]


def evaluate_predictions(
    pred: pd.DataFrame, truth_df: pd.DataFrame
) -> dict[str, float]:
    truth = truth_df[["id", "op", "target_id", "value"]].copy()
    truth["value"] = truth["value"].fillna("").astype(str)
    merged = pred.merge(truth, on="id", suffixes=("_pred", "_true"))
    target_ok = (
        merged["target_id_pred"].astype(str).eq(merged["target_id_true"].astype(str))
    )
    op_ok = merged["op_pred"].astype(str).eq(merged["op_true"].astype(str))
    value_ok = (
        merged["value_pred"]
        .fillna("")
        .astype(str)
        .eq(merged["value_true"].fillna("").astype(str))
    )
    return {
        "target_id": float(target_ok.mean()),
        "op": float(op_ok.mean()),
        "value": float(value_ok.mean()),
        "all_match": float((target_ok & op_ok & value_ok).mean()),
    }


def validate_submission(
    submission: pd.DataFrame, test_df: pd.DataFrame, sample_df: pd.DataFrame
) -> dict[str, Any]:
    test_by_id = {row["id"]: row for _, row in test_df.iterrows()}
    target_missing = 0
    for _, row in submission.iterrows():
        ids = {
            cand.get("candidate_id")
            for cand in load_candidates(test_by_id[row["id"]].get("candidate_elements"))
        }
        if row["target_id"] not in ids:
            target_missing += 1

    return {
        "rows": int(len(submission)),
        "ids_match_sample_order": submission["id"].tolist() == sample_df["id"].tolist(),
        "unique_ids": int(submission["id"].nunique()),
        "invalid_ops": int((~submission["op"].isin(["CLICK", "TYPE", "SELECT"])).sum()),
        "click_value_nonempty": int(
            (
                (submission["op"] == "CLICK")
                & submission["value"].fillna("").astype(str).str.strip().ne("")
            ).sum()
        ),
        "nonclick_value_empty": int(
            (
                (submission["op"] != "CLICK")
                & submission["value"].fillna("").astype(str).str.strip().eq("")
            ).sum()
        ),
        "target_id_not_in_candidates": int(target_missing),
        "op_dist": submission["op"].value_counts().to_dict(),
    }


def train_and_predict(output_path: Path, n_estimators: int = 900) -> None:
    print("[load] raw train/test")
    train_df = read_csv(DATA_DIR / "train.csv")
    test_df = read_csv(DATA_DIR / "test.csv")
    sample_df = read_csv(DATA_DIR / "sample_submission.csv")
    print(f"train rows={len(train_df)} test rows={len(test_df)}")

    print("[features] build raw candidate features")
    idf = compute_idf(train_df)
    train_feat = build_candidate_features(
        train_df, include_label=True, idf=idf
    ).sort_values(["row_idx", "pos"])
    test_feat = build_candidate_features(
        test_df, include_label=False, idf=idf
    ).sort_values(["row_idx", "pos"])

    feature_cols = [
        col
        for col in train_feat.columns
        if col not in {"row_idx", "id", "candidate_id", "label"}
    ]
    cat_cols = [
        "site",
        "tag",
        "type",
        "role",
        "name",
        "label_text",
        "text_key",
        "placeholder",
    ]

    row_ids = train_df.index.to_numpy()
    tr_rows, val_rows = train_test_split(
        row_ids, test_size=0.2, random_state=SEED, stratify=train_df["op"]
    )
    tr_set = set(tr_rows)
    val_set = set(val_rows)
    tr_feat = (
        train_feat[train_feat["row_idx"].isin(tr_set)]
        .copy()
        .sort_values(["row_idx", "pos"])
    )
    val_feat = (
        train_feat[train_feat["row_idx"].isin(val_set)]
        .copy()
        .sort_values(["row_idx", "pos"])
    )
    test_feat_for_val = test_feat.copy()
    cast_categories(tr_feat, val_feat, test_feat_for_val, cat_cols)

    print("[train] validation ranker")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
    )
    ranker.fit(
        tr_feat[feature_cols],
        tr_feat["label"].astype(int),
        group=group_sizes(tr_feat),
        eval_set=[(val_feat[feature_cols], val_feat["label"].astype(int))],
        eval_group=[group_sizes(val_feat)],
        eval_at=[1, 3, 5],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    val_scores = ranker.predict(
        val_feat[feature_cols], num_iteration=ranker.best_iteration_
    )
    val_targets = predict_targets(val_feat, val_scores)
    val_df = train_df.iloc[val_rows].copy()
    val_pred = attach_op_value(
        val_targets,
        val_df,
        build_candidate_map(val_df),
        build_op_tables(train_df.iloc[tr_rows]),
    )
    metrics = evaluate_predictions(val_pred, val_df)
    print(
        "[valid]",
        {key: round(value, 4) for key, value in metrics.items()},
        "best_iter",
        ranker.best_iteration_,
    )

    print("[train] final ranker on full train")
    full_feat = train_feat.copy().sort_values(["row_idx", "pos"])
    final_test_feat = test_feat.copy().sort_values(["row_idx", "pos"])
    cast_categories(full_feat, None, final_test_feat, cat_cols)
    final_iters = int(max(250, (ranker.best_iteration_ or 400) * 1.15))
    final_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=final_iters,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED + 1,
        verbose=-1,
    )
    final_ranker.fit(
        full_feat[feature_cols],
        full_feat["label"].astype(int),
        group=group_sizes(full_feat),
        categorical_feature=cat_cols,
    )

    print("[predict] test")
    test_scores = final_ranker.predict(final_test_feat[feature_cols])
    test_targets = predict_targets(final_test_feat, test_scores)
    submission = attach_op_value(
        test_targets, test_df, build_candidate_map(test_df), build_op_tables(train_df)
    )
    submission = sample_df[["id"]].merge(submission, on="id", how="left")
    submission["op"] = submission["op"].fillna("CLICK")
    submission["target_id"] = submission["target_id"].fillna("")
    submission["value"] = submission["value"].fillna("")
    submission.loc[submission["op"].eq("CLICK"), "value"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission[["id", "op", "target_id", "value"]].to_csv(
        output_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    print("[saved]", output_path)
    print("[check]", validate_submission(submission, test_df, sample_df))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT v8 raw train.csv LightGBM candidate-ranker submission builder."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=900)
    args = parser.parse_args()
    train_and_predict(args.output, n_estimators=args.n_estimators)


if __name__ == "__main__":
    main()
