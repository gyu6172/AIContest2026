# -*- coding: utf-8 -*-
"""preprocess.py에 새 함수들을 추가하는 헬퍼 스크립트"""
import os

NEW_CODE = r'''

# =============================================================
# HTML 타입 감지 및 Workflow 컨텍스트 파싱 (신규 추가)
# =============================================================

def is_workflow_html(html_str):
    """Workflow형 HTML 여부 판별."""
    if not isinstance(html_str, str):
        return False
    return ("workflow-context" in html_str) or ("completed-fields" in html_str)


def detect_html_type(row):
    """행의 cleaned_html을 보고 'workflow' 또는 'real_web' 반환."""
    return "workflow" if is_workflow_html(str(row.get("cleaned_html", ""))) else "real_web"


def extract_workflow_context(html_str):
    """Workflow HTML에서 현재 단계 및 완료된 필드 정보를 추출한다.

    반환 예시:
        {"current_step": "6", "total_steps": "7", "completed_fields": ["Name", "Date"]}
    """
    result = {"current_step": "", "total_steps": "", "completed_fields": []}
    if not isinstance(html_str, str):
        return result
    step_m = re.search(r"current step\s+(\d+)\s+of\s+(\d+)", html_str, re.IGNORECASE)
    if step_m:
        result["current_step"] = step_m.group(1)
        result["total_steps"]  = step_m.group(2)
    completed_m = re.search(r"Completed:\s*([^\n<]+)", html_str, re.IGNORECASE)
    if completed_m:
        fields = [f.strip() for f in completed_m.group(1).split(",") if f.strip()]
        result["completed_fields"] = fields
    return result


# =============================================================
# 번호 기반 후보 포맷 및 변환 (신규 추가)
# =============================================================

def format_numbered_candidates(candidates):
    """후보를 1~N 번호로 포맷팅한다.

    핵심 개선:
    - LLM이 긴 ID 문자열을 생성하는 대신 숫자(1~15)만 선택하면 됨
    - hallucination 및 형식 오류 대폭 감소
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        tag         = c.get("tag", "")
        text        = str(c.get("text", "")).strip()
        attrs       = str(c.get("attrs", "")).strip()
        attrs_short = (attrs[:120] + "...") if len(attrs) > 120 else attrs

        line = f"{i:>2}. [{tag}]"
        if text:
            line += f' "{text}"'
        if attrs_short:
            line += f" | {attrs_short}"
        lines.append(line)
    return "\n".join(lines)


def choice_to_candidate_id(choice, candidates):
    """번호(1~N)를 실제 candidate_id로 변환한다.

    LLM이 {"choice": 3} 을 출력하면 candidates[2]의 candidate_id를 반환.
    변환 실패 시 빈 문자열 반환 -> 호출자가 fallback 처리.
    """
    try:
        idx = int(choice) - 1  # 1-indexed -> 0-indexed
        if 0 <= idx < len(candidates):
            return str(candidates[idx].get("candidate_id", ""))
    except (ValueError, TypeError):
        pass
    return ""
'''

target = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "preprocess.py"
)
target = os.path.normpath(target)

with open(target, "a", encoding="utf-8") as f:
    f.write(NEW_CODE)

print(f"Done: appended to {target}")
