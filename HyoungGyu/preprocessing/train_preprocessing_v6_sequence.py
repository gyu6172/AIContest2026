from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "HyoungGyu" / "preprocessing").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/ and HyoungGyu/preprocessing/.")


ROOT = find_project_root(SCRIPT_DIR)
DATA_DIR = ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "train.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "preprocessed_train_v6_sequence.csv"


DOMAIN_STOP = {"task", "step", "enter", "action", "element"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOP

TOKEN_RE = re.compile(r"[A-Za-z0-9/._-]+")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
HISTORY_STEP_RE = re.compile(
    r"Step\s*(\d+)\s*:\s*\[([^\]]+)\]\s*(.*?)\s*->\s*([A-Za-z]+)(?::\s*(.*?))?(?=\n\s*Step\s*\d+\s*:|\Z)",
    re.I | re.S,
)
TAG_RE = re.compile(r"<[^>]+>")
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
WORKFLOW_CONTEXT_RE = re.compile(
    r'<aside[^>]*class=["\'][^"\']*\bworkflow-context\b[^"\']*["\'][^>]*>(.*?)</aside>',
    re.I | re.S,
)
COMPLETED_FIELDS_RE = re.compile(
    r'<aside[^>]*class=["\'][^"\']*\bcompleted-fields\b[^"\']*["\'][^>]*>(.*?)</aside>',
    re.I | re.S,
)
CURRENT_STEP_RE = re.compile(r"current\s+step\s+(\d+)\s+of\s+(\d+)", re.I)
PANEL_RE = re.compile(
    r'<section[^>]*aria-label=["\']current workflow panel["\'][^>]*>(.*?)</section>',
    re.I | re.S,
)
LABEL_RE = re.compile(r"<label[^>]*>(.*?)</label>", re.I | re.S)
OPENING_TAG_RE = re.compile(r"<(input|select|button|a|textarea)\b([^>]*)>", re.I | re.S)
HTML_ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*["\']([^"\']*)["\']')
PIPE_ATTR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_-]*)=([^|]+?)(?=\s*\||$)", re.I)
OPTIONS_RE = re.compile(r"\boptions=([^|]+?)(?=\s*\||$)", re.I)
TYPE_ATTR_RE = re.compile(r"\btype=([^|]+?)(?=\s*\||$)", re.I)

CLICK_INPUT_TYPES = {"checkbox", "radio", "button", "submit", "image", "range"}
TYPE_INPUT_TYPES = {"text", "date", "email", "number", "password", "tel", "url", "time", "search"}


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def as_text(value) -> str:
    if is_missing(value):
        return ""
    return str(value)


def normalize_space(value: str) -> str:
    value = html.unescape(as_text(value))
    return re.sub(r"\s+", " ", value).strip()


def strip_tags(value: str) -> str:
    return normalize_space(TAG_RE.sub(" ", as_text(value)))


def split_field_text(value: str) -> str:
    value = as_text(value).replace("_", " ")
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[^A-Za-z0-9]+", " ", value).strip()


def norm_key(value: str) -> str:
    value = split_field_text(value).lower()
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value) -> list[str]:
    text = as_text(value)
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def clean_text(value) -> str:
    return " ".join(tokenize(value))


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def safe_json_loads(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def classify_value_type(value: str) -> str:
    value = normalize_space(value)
    if not value:
        return "blank"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "iso_date"
    if re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+", value):
        return "email"
    if re.fullmatch(r"[A-Z]{2,6}-[A-Z0-9]{2,10}", value):
        return "code"
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return "number"
    if re.fullmatch(r"[A-Z]{2,5}", value):
        return "short_code"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", value):
        return "title_phrase"
    return "text"


def parse_history_steps(history_value) -> list[dict]:
    history_text = as_text(history_value).replace("\r\n", "\n").replace("\r", "\n")
    if not history_text.strip():
        return []

    steps = []
    for match in HISTORY_STEP_RE.finditer(history_text):
        step_no = int(match.group(1))
        tag = normalize_space(match.group(2)).lower()
        element_text = normalize_space(match.group(3))
        op = normalize_space(match.group(4)).upper()
        value = normalize_space(match.group(5) or "")
        steps.append(
            {
                "step_no": step_no,
                "tag": tag,
                "text": element_text,
                "text_key": norm_key(element_text),
                "op": op,
                "value": value,
                "value_type": classify_value_type(value),
            }
        )
    return steps


def parse_pipe_attrs(attrs_value) -> dict[str, str]:
    attrs = as_text(attrs_value)
    out = {}
    for match in PIPE_ATTR_RE.finditer(attrs):
        key = match.group(1).strip().lower()
        value = normalize_space(match.group(2))
        if key and value:
            out[key] = value
    return out


def parse_options(attrs_value) -> list[str]:
    attrs = as_text(attrs_value)
    match = OPTIONS_RE.search(attrs)
    if not match:
        return []
    return [normalize_space(part) for part in match.group(1).split("/") if normalize_space(part)]


def input_type_value(attrs_value) -> str:
    match = TYPE_ATTR_RE.search(as_text(attrs_value))
    return normalize_space(match.group(1)).lower() if match else ""


def predict_candidate_op(candidate: dict) -> str:
    tag = as_text(candidate.get("tag")).lower()
    attrs = as_text(candidate.get("attrs"))
    attr_dict = parse_pipe_attrs(attrs)
    input_type = input_type_value(attrs)
    role = attr_dict.get("role", "").lower()

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


def candidate_field_key(candidate: dict, attr_dict: dict[str, str] | None = None) -> str:
    attr_dict = attr_dict or parse_pipe_attrs(candidate.get("attrs"))
    for value in [
        attr_dict.get("name", ""),
        attr_dict.get("label", ""),
        as_text(candidate.get("text")),
        attr_dict.get("placeholder", ""),
    ]:
        key = norm_key(value)
        if key:
            return key
    return ""


def parse_html_attrs(attr_text: str) -> dict[str, str]:
    return {match.group(1).lower(): html.unescape(match.group(2)) for match in HTML_ATTR_RE.finditer(attr_text)}


def parse_html_context(html_value) -> dict:
    raw_html = as_text(html_value)
    if not raw_html:
        return {
            "h1_text": "",
            "workflow_context": "",
            "current_step": 0,
            "total_steps": 0,
            "completed_fields": [],
            "panel_names": [],
            "panel_labels": [],
            "panel_text": "",
            "panel_controls": [],
            "html_text_clean": "",
        }

    h1_text = " ".join(strip_tags(match.group(1)) for match in H1_RE.finditer(raw_html)).strip()
    workflow_context = " ".join(strip_tags(match.group(1)) for match in WORKFLOW_CONTEXT_RE.finditer(raw_html)).strip()

    current_step = 0
    total_steps = 0
    step_match = CURRENT_STEP_RE.search(workflow_context) or CURRENT_STEP_RE.search(raw_html)
    if step_match:
        current_step = int(step_match.group(1))
        total_steps = int(step_match.group(2))

    completed_fields = []
    for match in COMPLETED_FIELDS_RE.finditer(raw_html):
        completed_text = strip_tags(match.group(1))
        completed_text = re.sub(r"^Completed:\s*", "", completed_text, flags=re.I)
        for part in completed_text.split(","):
            item = normalize_space(part)
            if item and item.lower() != "none":
                completed_fields.append(item)

    panel_html = ""
    panel_match = PANEL_RE.search(raw_html)
    if panel_match:
        panel_html = panel_match.group(1)

    panel_names = []
    panel_controls = []
    if panel_html:
        for tag_match in OPENING_TAG_RE.finditer(panel_html):
            tag = tag_match.group(1).lower()
            attrs = parse_html_attrs(tag_match.group(2))
            name = normalize_space(attrs.get("name", ""))
            if name:
                panel_names.append(name)
            panel_controls.append(
                {
                    "tag": tag,
                    "name": name,
                    "type": normalize_space(attrs.get("type", "")).lower(),
                    "role": normalize_space(attrs.get("role", "")).lower(),
                    "aria_label": normalize_space(attrs.get("aria-label", "")),
                    "placeholder": normalize_space(attrs.get("placeholder", "")),
                }
            )

    panel_labels = []
    if panel_html:
        for label_match in LABEL_RE.finditer(panel_html):
            label_html = re.sub(r"<(input|select|textarea|button|a)\b.*", "", label_match.group(1), flags=re.I | re.S)
            label_text = strip_tags(label_html)
            if label_text:
                panel_labels.append(label_text)

    panel_text = strip_tags(panel_html)
    html_text_clean = clean_text(strip_tags(raw_html))

    def dedupe(values: list[str]) -> list[str]:
        out = []
        seen = set()
        for value in values:
            key = norm_key(value)
            if key and key not in seen:
                seen.add(key)
                out.append(value)
        return out

    return {
        "h1_text": h1_text,
        "workflow_context": workflow_context,
        "current_step": current_step,
        "total_steps": total_steps,
        "completed_fields": dedupe(completed_fields),
        "panel_names": dedupe(panel_names),
        "panel_labels": dedupe(panel_labels),
        "panel_text": panel_text,
        "panel_controls": panel_controls,
        "html_text_clean": html_text_clean,
    }


def enrich_candidates(
    candidates_value,
    task_tokens: set[str],
    remaining_tokens: set[str],
    history_completed_keys: set[str],
    html_completed_keys: set[str],
    panel_name_keys: set[str],
    panel_label_keys: set[str],
) -> list[dict]:
    candidates = safe_json_loads(candidates_value)
    enriched = []

    for pos, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        attrs = as_text(candidate.get("attrs"))
        attr_dict = parse_pipe_attrs(attrs)
        tag = as_text(candidate.get("tag")).lower()
        text = normalize_space(candidate.get("text"))
        label = attr_dict.get("label", "")
        name = attr_dict.get("name", "")
        placeholder = attr_dict.get("placeholder", "")
        field_key = candidate_field_key(candidate, attr_dict)
        candidate_tokens = set(tokenize(" ".join([text, attrs])))
        input_type = input_type_value(attrs)
        options = parse_options(attrs)

        name_key = norm_key(name)
        label_key = norm_key(label)
        text_key = norm_key(text)
        keys = {key for key in [field_key, name_key, label_key, text_key] if key}

        enriched.append(
            {
                "candidate_id": as_text(candidate.get("candidate_id")),
                "pos": pos,
                "tag": tag,
                "text": text,
                "field_key": field_key,
                "name": name,
                "label": label,
                "placeholder": placeholder,
                "input_type": input_type,
                "role": attr_dict.get("role", "").lower(),
                "predicted_op": predict_candidate_op(candidate),
                "n_options": len(options),
                "options": options,
                "n_tokens": len(candidate_tokens),
                "task_overlap": len(task_tokens & candidate_tokens),
                "remaining_overlap": len(remaining_tokens & candidate_tokens),
                "in_history_completed": int(bool(keys & history_completed_keys)),
                "in_html_completed": int(bool(keys & html_completed_keys)),
                "in_workflow_panel": int(bool((name_key and name_key in panel_name_keys) or (label_key and label_key in panel_label_keys))),
            }
        )
    return enriched


def preprocess_row(row: pd.Series) -> dict:
    task = as_text(row.get("task"))
    history = as_text(row.get("history"))
    cleaned_html = as_text(row.get("cleaned_html"))

    task_tokens = set(tokenize(task))
    history_tokens = set(tokenize(history))
    remaining_tokens = task_tokens - history_tokens

    history_steps = parse_history_steps(history)
    history_completed_keys = {step["text_key"] for step in history_steps if step.get("text_key")}
    last_step = history_steps[-1] if history_steps else {}

    html_context = parse_html_context(cleaned_html)
    html_completed_keys = {norm_key(value) for value in html_context["completed_fields"] if norm_key(value)}
    panel_name_keys = {norm_key(value) for value in html_context["panel_names"] if norm_key(value)}
    panel_label_keys = {norm_key(value) for value in html_context["panel_labels"] if norm_key(value)}

    candidates_enriched = enrich_candidates(
        row.get("candidate_elements"),
        task_tokens,
        remaining_tokens,
        history_completed_keys,
        html_completed_keys,
        panel_name_keys,
        panel_label_keys,
    )
    target_id = as_text(row.get("target_id"))
    target_candidate = next((candidate for candidate in candidates_enriched if candidate["candidate_id"] == target_id), {})

    next_step_no = len(history_steps) + 1
    html_current_step = html_context["current_step"]
    html_total_steps = html_context["total_steps"]

    return {
        "task_clean": clean_text(task),
        "history_clean": clean_text(history),
        "html_text_clean": html_context["html_text_clean"],
        "task_tokens_json": json_dumps(sorted(task_tokens)),
        "history_tokens_json": json_dumps(sorted(history_tokens)),
        "remaining_task_tokens_json": json_dumps(sorted(remaining_tokens)),
        "history_steps_json": json_dumps(history_steps),
        "n_history_steps": len(history_steps),
        "last_step_no": last_step.get("step_no", 0),
        "last_step_tag": last_step.get("tag", ""),
        "last_step_text": last_step.get("text", ""),
        "last_step_op": last_step.get("op", ""),
        "last_step_value": last_step.get("value", ""),
        "last_step_value_type": last_step.get("value_type", ""),
        "history_step_texts_json": json_dumps([step["text"] for step in history_steps]),
        "history_step_ops_json": json_dumps([step["op"] for step in history_steps]),
        "history_completed_keys_json": json_dumps(sorted(history_completed_keys)),
        "html_current_step": html_current_step,
        "html_total_steps": html_total_steps,
        "html_step_remaining": max(0, html_total_steps - html_current_step) if html_total_steps else 0,
        "html_h1_text": html_context["h1_text"],
        "html_workflow_context": html_context["workflow_context"],
        "html_completed_fields_json": json_dumps(html_context["completed_fields"]),
        "html_panel_names_json": json_dumps(html_context["panel_names"]),
        "html_panel_labels_json": json_dumps(html_context["panel_labels"]),
        "html_panel_controls_json": json_dumps(html_context["panel_controls"]),
        "html_panel_text": html_context["panel_text"],
        "next_step_no": next_step_no,
        "step_alignment_delta": html_current_step - next_step_no if html_current_step else 0,
        "candidate_enriched_json": json_dumps(candidates_enriched),
        "n_candidates": len(candidates_enriched),
        "target_candidate_pos": target_candidate.get("pos", -1),
        "target_candidate_tag": target_candidate.get("tag", ""),
        "target_candidate_predicted_op": target_candidate.get("predicted_op", ""),
        "target_in_history_completed": target_candidate.get("in_history_completed", 0),
        "target_in_html_completed": target_candidate.get("in_html_completed", 0),
        "target_in_workflow_panel": target_candidate.get("in_workflow_panel", 0),
    }


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8", encoding_errors="replace")
    records = [preprocess_row(row) for _, row in df.iterrows()]
    extra = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), extra], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequence-aware preprocessing from raw train.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    out = preprocess(args.input, args.output)

    parsed_history_rate = float((out["n_history_steps"] > 0).mean())
    html_step_rate = float((out["html_current_step"] > 0).mean())
    has_html_step = out["html_current_step"] > 0
    alignment_rate = float((out.loc[has_html_step, "step_alignment_delta"] == 0).mean()) if has_html_step.any() else 0.0
    target_in_panel_rate = float(out.loc[has_html_step, "target_in_workflow_panel"].mean()) if has_html_step.any() else 0.0

    print("rows:", len(out))
    print("columns:", len(out.columns))
    print(f"history parsed rate: {parsed_history_rate:.4f}")
    print(f"html current-step rate: {html_step_rate:.4f}")
    print(f"history/html step alignment rate: {alignment_rate:.4f}")
    print(f"target in workflow panel rate: {target_in_panel_rate:.4f}")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
