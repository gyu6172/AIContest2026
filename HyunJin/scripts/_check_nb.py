# -*- coding: utf-8 -*-
import json
nb = json.load(open(r'c:\Users\shjsh\Desktop\ai contest\HyunJin\colab_pipeline.ipynb', encoding='utf-8'))
print(f"cells: {len(nb['cells'])}, nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
for i, c in enumerate(nb['cells']):
    src = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f"  [{i+1:>2}] {c['cell_type']:8} | {src[:55]}")
