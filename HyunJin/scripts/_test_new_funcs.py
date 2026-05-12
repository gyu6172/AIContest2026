# -*- coding: utf-8 -*-
import sys, json
sys.path.insert(0, r'c:\Users\shjsh\Desktop\ai contest\HyunJin\src')
from preprocess import (
    detect_html_type, extract_workflow_context,
    format_numbered_candidates, choice_to_candidate_id
)

# 1. HTML 타입 감지
row_wf = {"cleaned_html": "blah workflow-context blah current step 3 of 7 blah"}
row_rw = {"cleaned_html": "<html><body><button>Submit</button></body></html>"}
print("HTML type (workflow):", detect_html_type(row_wf))
print("HTML type (real_web):", detect_html_type(row_rw))

# 2. Workflow 컨텍스트 파싱
html_wf = "workflow-context current step 3 of 7 Completed: Name, Date, Email"
ctx = extract_workflow_context(html_wf)
print("Workflow ctx:", ctx)

# 3. 번호 포맷
cands = [
    {"candidate_id": "c_1", "tag": "input",  "text": "Username", "attrs": "placeholder=Enter name"},
    {"candidate_id": "c_2", "tag": "button", "text": "Submit",   "attrs": ""},
    {"candidate_id": "c_3", "tag": "select", "text": "Country",  "attrs": "options=USA / UK / Canada"},
]
print()
print(format_numbered_candidates(cands))

# 4. 번호 -> ID 변환
print()
print("choice 1 ->", choice_to_candidate_id(1, cands))
print("choice 2 ->", choice_to_candidate_id(2, cands))
print("choice 99 ->", repr(choice_to_candidate_id(99, cands)))
