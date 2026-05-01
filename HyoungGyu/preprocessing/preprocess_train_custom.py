from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "to",
            "with",
        }
    )


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "HyoungGyu" / "preprocessing").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/ and HyoungGyu/preprocessing/.")


ROOT = find_project_root(SCRIPT_DIR)
DATA_DIR = ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "train.csv"
DEFAULT_ROW_OUTPUT = SCRIPT_DIR / "preprocessed_train_custom.csv"
DEFAULT_CANDIDATE_OUTPUT = SCRIPT_DIR / "preprocessed_train_candidates_custom.csv"

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[A-Za-z0-9/._:+,-]*[A-Za-z0-9])?")
HISTORY_STEP_RE = re.compile(
    r"Step\s*(\d+)\s*:\s*\[([^\]]+)\]\s*(.*?)\s*->\s*([A-Za-z]+)(?::\s*(.*?))?(?=\n\s*Step\s*\d+\s*:|\Z)",
    re.I | re.S,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
WORKFLOW_HINT_RE = re.compile(
    r"workflow-context|completed-fields|current workflow panel|current\s+step\s+\d+\s+of\s+\d+",
    re.I,
)
CURRENT_STEP_RE = re.compile(r"current\s+step\s+(\d+)\s+of\s+(\d+)", re.I)
WORKFLOW_CONTEXT_RE = re.compile(
    r'<aside[^>]*class=["\'][^"\']*\bworkflow-context\b[^"\']*["\'][^>]*>(.*?)</aside>',
    re.I | re.S,
)
COMPLETED_FIELDS_RE = re.compile(
    r'<aside[^>]*class=["\'][^"\']*\bcompleted-fields\b[^"\']*["\'][^>]*>(.*?)</aside>',
    re.I | re.S,
)
PANEL_RE = re.compile(
    r'<section[^>]*aria-label=["\']current workflow panel["\'][^>]*>(.*?)</section>',
    re.I | re.S,
)
LABEL_RE = re.compile(r"<label[^>]*>(.*?)</label>", re.I | re.S)
OPENING_TAG_RE = re.compile(r"<(input|select|button|a|textarea)\b([^>]*)>", re.I | re.S)
HTML_ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*["\']([^"\']*)["\']')
PIPE_ATTR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*)=([^|]*?)(?=\s*\||$)", re.I)

CLICK_INPUT_TYPES = {"button", "checkbox", "image", "radio", "range", "reset", "submit"}
TYPE_INPUT_TYPES = {"date", "email", "number", "password", "search", "tel", "text", "time", "url"}


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def as_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(as_text(value))).strip()


def strip_tags(value: Any) -> str:
    return normalize_space(HTML_TAG_RE.sub(" ", as_text(value)))


def split_field_text(value: Any) -> str:
    text = as_text(value).replace("_", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"[^A-Za-z0-9]+", " ", text).strip()


def norm_key(value: Any) -> str:
    return re.sub(r"\s+", " ", split_field_text(value).lower()).strip()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_json_loads(value: Any) -> list[dict]:
    if isinstance(value, list):
        return value
    raw = as_text(value)
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def tokenize_preserve_case(value: Any) -> list[str]:
    tokens = TOKEN_RE.findall(as_text(value))
    return [token for token in tokens if token.lower() not in STOPWORDS]


def clean_token_text(value: Any) -> str:
    return " ".join(tokenize_preserve_case(value))


def clean_site_token(value: Any) -> tuple[str, int]:
    site = as_text(value).strip()
    if site.startswith("site_"):
        return site[5:], 1
    return site, 0


def clean_task(value: Any) -> tuple[str, list[str], int]:
    raw = as_text(value)
    cleaned, n_subs = re.subn(r"^\s*Task\s*:\s*", "", raw, count=1, flags=re.I)
    tokens = tokenize_preserve_case(cleaned)
    return " ".join(tokens), tokens, int(n_subs > 0)


def classify_html_type(cleaned_html: Any) -> str:
    return "workflow" if WORKFLOW_HINT_RE.search(as_text(cleaned_html)) else "raw_web"


def parse_html_attrs(attr_text: str) -> dict[str, str]:
    return {match.group(1).lower(): normalize_space(match.group(2)) for match in HTML_ATTR_RE.finditer(attr_text)}


def parse_workflow_html(cleaned_html: Any) -> dict[str, Any]:
    raw_html = as_text(cleaned_html)
    workflow_context = " ".join(strip_tags(match.group(1)) for match in WORKFLOW_CONTEXT_RE.finditer(raw_html))
    h1_text = " ".join(strip_tags(match.group(1)) for match in H1_RE.finditer(raw_html))

    step_match = CURRENT_STEP_RE.search(workflow_context) or CURRENT_STEP_RE.search(raw_html)
    current_step = int(step_match.group(1)) if step_match else 0
    total_steps = int(step_match.group(2)) if step_match else 0

    completed_fields: list[str] = []
    for match in COMPLETED_FIELDS_RE.finditer(raw_html):
        completed_text = re.sub(r"^Completed:\s*", "", strip_tags(match.group(1)), flags=re.I)
        for part in completed_text.split(","):
            item = normalize_space(part)
            if item and item.lower() != "none":
                completed_fields.append(item)

    panel_html = ""
    panel_match = PANEL_RE.search(raw_html)
    if panel_match:
        panel_html = panel_match.group(1)

    panel_names: list[str] = []
    panel_labels: list[str] = []
    panel_controls: list[dict[str, str]] = []

    if panel_html:
        for tag_match in OPENING_TAG_RE.finditer(panel_html):
            tag = tag_match.group(1).lower()
            attrs = parse_html_attrs(tag_match.group(2))
            name = attrs.get("name", "")
            if name:
                panel_names.append(name)
            panel_controls.append(
                {
                    "tag": tag,
                    "name": name,
                    "type": attrs.get("type", "").lower(),
                    "role": attrs.get("role", "").lower(),
                    "placeholder": attrs.get("placeholder", ""),
                    "aria_label": attrs.get("aria-label", ""),
                }
            )

        for label_match in LABEL_RE.finditer(panel_html):
            label_html = re.sub(
                r"<(input|select|textarea|button|a)\b.*",
                "",
                label_match.group(1),
                flags=re.I | re.S,
            )
            label_text = strip_tags(label_html)
            if label_text:
                panel_labels.append(label_text)

    return {
        "current_step": current_step,
        "total_steps": total_steps,
        "h1_text": normalize_space(h1_text),
        "workflow_context": normalize_space(workflow_context),
        "completed_fields": dedupe_preserve_order(completed_fields),
        "panel_names": dedupe_preserve_order(panel_names),
        "panel_labels": dedupe_preserve_order(panel_labels),
        "panel_controls": panel_controls,
        "panel_text": strip_tags(panel_html),
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = norm_key(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def classify_value_type(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return "blank"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "iso_date"
    if re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return "email"
    if re.fullmatch(r"[A-Z]{2,6}-[A-Z0-9]{2,10}", text):
        return "code"
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return "number"
    if re.fullmatch(r"[A-Z]{2,5}", text):
        return "short_code"
    return "text"


def parse_history_steps(history_value: Any) -> list[dict[str, Any]]:
    history_text = as_text(history_value).replace("\r\n", "\n").replace("\r", "\n")
    if not history_text.strip():
        return []

    steps: list[dict[str, Any]] = []
    for match in HISTORY_STEP_RE.finditer(history_text):
        step_no = int(match.group(1))
        tag = normalize_space(match.group(2)).lower()
        text = normalize_space(match.group(3))
        op = normalize_space(match.group(4)).upper()
        value = normalize_space(match.group(5) or "")
        steps.append(
            {
                "step_no": step_no,
                "tag": tag,
                "text": text,
                "text_key": norm_key(text),
                "op": op,
                "value": value,
                "value_type": classify_value_type(value),
            }
        )
    return steps


def parse_pipe_attrs(attrs_value: Any) -> dict[str, str]:
    attrs = as_text(attrs_value)
    out: dict[str, str] = {}
    for match in PIPE_ATTR_RE.finditer(attrs):
        key = match.group(1).strip().lower()
        value = normalize_space(match.group(2))
        if key:
            out[key] = value
    return out


def parse_options(attr_dict: dict[str, str]) -> list[str]:
    options = attr_dict.get("options", "")
    if not options:
        return []
    return [normalize_space(part) for part in options.split("/") if normalize_space(part)]


def predicted_candidate_op(tag: str, attr_dict: dict[str, str]) -> str:
    tag = tag.lower()
    input_type = attr_dict.get("type", "").lower()
    role = attr_dict.get("role", "").lower()
    if tag == "select":
        return "SELECT"
    if tag == "textarea":
        return "TYPE"
    if tag == "input":
        if input_type in CLICK_INPUT_TYPES or role == "button":
            return "CLICK"
        return "TYPE"
    if tag in {"a", "button", "link"}:
        return "CLICK"
    return "CLICK"


def candidate_field_key(candidate: dict[str, Any], attr_dict: dict[str, str]) -> str:
    for value in [
        attr_dict.get("name", ""),
        attr_dict.get("label", ""),
        as_text(candidate.get("text")),
        attr_dict.get("placeholder", ""),
        attr_dict.get("aria_label", ""),
        attr_dict.get("aria-label", ""),
        attr_dict.get("title", ""),
        attr_dict.get("alt", ""),
    ]:
        key = norm_key(value)
        if key:
            return key
    return ""


def normalize_candidate(candidate: dict[str, Any], idx: int, tag_counts: Counter[str]) -> dict[str, Any]:
    candidate_id = as_text(candidate.get("candidate_id"))
    tag_raw = as_text(candidate.get("tag")).strip()
    tag = tag_raw.lower()
    text = normalize_space(candidate.get("text"))
    attrs_raw = normalize_space(candidate.get("attrs"))
    attr_dict = parse_pipe_attrs(attrs_raw)
    options = parse_options(attr_dict)
    text_tokens = tokenize_preserve_case(text)
    attrs_tokens = tokenize_preserve_case(attrs_raw)
    tag_count = tag_counts.get(tag, 0)

    return {
        "candidate_idx": idx,
        "candidate_id": candidate_id,
        "candidate_tag": tag,
        "candidate_tag_raw": tag_raw,
        "candidate_tag_count_in_row": tag_count,
        "candidate_tag_frac_in_row": tag_count / 15.0,
        "candidate_text": text,
        "candidate_attrs": attrs_raw,
        "candidate_attrs_json": attr_dict,
        "candidate_role": attr_dict.get("role", "").lower(),
        "candidate_type": attr_dict.get("type", "").lower(),
        "candidate_name": attr_dict.get("name", ""),
        "candidate_label": attr_dict.get("label", ""),
        "candidate_placeholder": attr_dict.get("placeholder", ""),
        "candidate_options": options,
        "candidate_options_count": len(options),
        "candidate_field_key": candidate_field_key(candidate, attr_dict),
        "candidate_text_tokens": text_tokens,
        "candidate_attrs_tokens": attrs_tokens,
        "candidate_text_empty": int(text == ""),
        "candidate_attrs_empty": int(attrs_raw == ""),
        "candidate_predicted_op": predicted_candidate_op(tag, attr_dict),
    }


def build_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    row_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "candidate_json_failures": 0,
        "site_prefix_removed": 0,
        "task_prefix_removed": 0,
        "tag_empty_candidates": 0,
    }

    has_target = "target_id" in df.columns
    has_op = "op" in df.columns
    has_value = "value" in df.columns

    for _, row in df.iterrows():
        row_id = as_text(row.get("id"))
        site_token_raw = as_text(row.get("site_token"))
        site_key, site_prefix_removed = clean_site_token(site_token_raw)
        stats["site_prefix_removed"] += site_prefix_removed

        task_raw = as_text(row.get("task"))
        task_clean, task_tokens, task_prefix_removed = clean_task(task_raw)
        stats["task_prefix_removed"] += task_prefix_removed

        history_raw = as_text(row.get("history"))
        history_steps = parse_history_steps(history_raw)
        last_step = history_steps[-1] if history_steps else {}

        cleaned_html_raw = as_text(row.get("cleaned_html"))
        html_type = classify_html_type(cleaned_html_raw)
        html_context = parse_workflow_html(cleaned_html_raw)

        candidates_raw = as_text(row.get("candidate_elements"))
        candidates = safe_json_loads(candidates_raw)
        if candidates_raw.strip() and not candidates:
            stats["candidate_json_failures"] += 1

        tag_counts = Counter(as_text(candidate.get("tag")).strip().lower() for candidate in candidates if isinstance(candidate, dict))
        stats["tag_empty_candidates"] += tag_counts.get("", 0)

        normalized_candidates = [
            normalize_candidate(candidate, idx, tag_counts)
            for idx, candidate in enumerate(candidates)
            if isinstance(candidate, dict)
        ]

        target_id = as_text(row.get("target_id")) if has_target else ""
        target_candidate = next(
            (candidate for candidate in normalized_candidates if candidate["candidate_id"] == target_id),
            {},
        )
        target_tag = as_text(target_candidate.get("candidate_tag"))
        target_tag_count = int(target_candidate.get("candidate_tag_count_in_row", 0) or 0)

        row_record: dict[str, Any] = {
            "id": row_id,
            "site_token_raw": site_token_raw,
            "site_key": site_key,
            "task_raw": task_raw,
            "task_clean": task_clean,
            "task_tokens_json": json_dumps(task_tokens),
            "history_raw": history_raw,
            "history_steps_json": json_dumps(history_steps),
            "history_step_count": len(history_steps),
            "history_last_step_no": last_step.get("step_no", 0),
            "history_last_tag": last_step.get("tag", ""),
            "history_last_text": last_step.get("text", ""),
            "history_last_op": last_step.get("op", ""),
            "history_last_value": last_step.get("value", ""),
            "history_ops_json": json_dumps([step["op"] for step in history_steps]),
            "history_tags_json": json_dumps([step["tag"] for step in history_steps]),
            "history_texts_json": json_dumps([step["text"] for step in history_steps]),
            "cleaned_html_raw": cleaned_html_raw,
            "html_type": html_type,
            "is_workflow": int(html_type == "workflow"),
            "html_current_step": html_context["current_step"],
            "html_total_steps": html_context["total_steps"],
            "html_step_remaining": max(0, html_context["total_steps"] - html_context["current_step"])
            if html_context["total_steps"]
            else 0,
            "html_h1_text": html_context["h1_text"],
            "html_workflow_context": html_context["workflow_context"],
            "html_completed_fields_json": json_dumps(html_context["completed_fields"]),
            "html_panel_names_json": json_dumps(html_context["panel_names"]),
            "html_panel_labels_json": json_dumps(html_context["panel_labels"]),
            "html_panel_controls_json": json_dumps(html_context["panel_controls"]),
            "html_panel_text": html_context["panel_text"],
            "candidate_elements_raw": candidates_raw,
            "candidate_elements_normalized_json": json_dumps(normalized_candidates),
            "candidate_tag_counts_json": json_dumps(dict(sorted(tag_counts.items()))),
            "candidate_unique_tag_count": len(tag_counts),
            "candidate_max_same_tag_count": max(tag_counts.values()) if tag_counts else 0,
            "n_candidates": len(normalized_candidates),
            "target_id": target_id,
            "target_tag": target_tag,
            "target_tag_count_in_candidates": target_tag_count,
            "target_tag_is_unique": int(target_tag_count == 1) if target_tag else 0,
        }
        if has_op:
            row_record["op"] = as_text(row.get("op"))
        if has_value:
            row_record["value"] = as_text(row.get("value"))
        row_records.append(row_record)

        for candidate in normalized_candidates:
            candidate_record = {
                "id": row_id,
                "site_token_raw": site_token_raw,
                "site_key": site_key,
                "task_clean": task_clean,
                "task_tokens_json": json_dumps(task_tokens),
                "history_step_count": len(history_steps),
                "history_last_op": last_step.get("op", ""),
                "html_type": html_type,
                "is_workflow": int(html_type == "workflow"),
                "html_current_step": html_context["current_step"],
                "html_total_steps": html_context["total_steps"],
                "candidate_idx": candidate["candidate_idx"],
                "candidate_id": candidate["candidate_id"],
                "candidate_tag": candidate["candidate_tag"],
                "candidate_tag_raw": candidate["candidate_tag_raw"],
                "candidate_tag_count_in_row": candidate["candidate_tag_count_in_row"],
                "candidate_tag_frac_in_row": candidate["candidate_tag_frac_in_row"],
                "candidate_text": candidate["candidate_text"],
                "candidate_attrs": candidate["candidate_attrs"],
                "candidate_attrs_json": json_dumps(candidate["candidate_attrs_json"]),
                "candidate_role": candidate["candidate_role"],
                "candidate_type": candidate["candidate_type"],
                "candidate_name": candidate["candidate_name"],
                "candidate_label": candidate["candidate_label"],
                "candidate_placeholder": candidate["candidate_placeholder"],
                "candidate_options_json": json_dumps(candidate["candidate_options"]),
                "candidate_options_count": candidate["candidate_options_count"],
                "candidate_field_key": candidate["candidate_field_key"],
                "candidate_text_tokens_json": json_dumps(candidate["candidate_text_tokens"]),
                "candidate_attrs_tokens_json": json_dumps(candidate["candidate_attrs_tokens"]),
                "candidate_text_empty": candidate["candidate_text_empty"],
                "candidate_attrs_empty": candidate["candidate_attrs_empty"],
                "candidate_predicted_op": candidate["candidate_predicted_op"],
                "target_id": target_id,
                "target_tag": target_tag,
                "label_is_target": int(candidate["candidate_id"] == target_id) if target_id else 0,
                "label_same_as_target_tag": int(candidate["candidate_tag"] == target_tag) if target_tag else 0,
            }
            if has_op:
                candidate_record["op"] = as_text(row.get("op"))
            if has_value:
                candidate_record["value"] = as_text(row.get("value"))
            candidate_records.append(candidate_record)

    return pd.DataFrame(row_records), pd.DataFrame(candidate_records), stats


def preprocess(input_path: Path, row_output: Path, candidate_output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(input_path, encoding="utf-8", encoding_errors="replace")
    row_df, candidate_df, stats = build_records(df)
    row_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(row_output, index=False, encoding="utf-8", lineterminator="\n")
    candidate_df.to_csv(candidate_output, index=False, encoding="utf-8", lineterminator="\n")
    return row_df, candidate_df, stats


def print_report(row_df: pd.DataFrame, candidate_df: pd.DataFrame, stats: dict[str, Any], row_output: Path, candidate_output: Path) -> None:
    print("rows:", len(row_df))
    print("candidate rows:", len(candidate_df))
    print("site_ prefix removed:", stats["site_prefix_removed"])
    print("Task: prefix removed:", stats["task_prefix_removed"])
    print("candidate JSON failures:", stats["candidate_json_failures"])
    print("empty candidate_tag:", stats["tag_empty_candidates"])
    print("html_type counts:", row_df["html_type"].value_counts(dropna=False).to_dict())
    print("history parsed rate:", round(float((row_df["history_step_count"] > 0).mean()), 4))
    if "target_tag" in row_df.columns and row_df["target_tag"].astype(str).str.len().sum() > 0:
        print("target_tag top10:", row_df["target_tag"].value_counts().head(10).to_dict())
        print("target tag unique rate:", round(float((row_df["target_tag_is_unique"] == 1).mean()), 4))
        print("candidate label_is_target sum:", int(candidate_df["label_is_target"].sum()))
    print("row output:", row_output)
    print("candidate output:", candidate_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess train.csv into row-level and candidate-level files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--row-output", type=Path, default=DEFAULT_ROW_OUTPUT)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    args = parser.parse_args()

    row_df, candidate_df, stats = preprocess(args.input, args.row_output, args.candidate_output)
    print_report(row_df, candidate_df, stats, args.row_output, args.candidate_output)


if __name__ == "__main__":
    main()
