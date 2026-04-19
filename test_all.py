#!/usr/bin/env python
"""测试所有 day 文件"""
import os
os.environ['OPENAI_API_KEY'] = 'sk-wtxvkihbrngdqdjlldbudlfneppbxqbuotsnwrdceshuaxdb'
os.environ['OPENAI_MODEL'] = 'Pro/MiniMaxAI/MiniMax-M2.5'

import subprocess
import sys

days = [
    'api/day1_components.py',
    'api/day2_model_io.py',
    'api/day3_retrieval.py',
    'api/day4_rag.py',
    'api/day5_agent.py',
    'api/day6_memory_chain.py',
    'api/day7_review.py',
]

for day in days:
    print(f"\n{'='*50}")
    print(f"Running {day}")
    print('='*50)
    result = subprocess.run(['uv', 'run', 'python', day], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
