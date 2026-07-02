#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __ver1_pathfix__  (정리: 데이터 경로가 원래대로 맞도록 작업폴더를 code/ 로 고정)
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

"""
_apply_pathfix.py — 1회용 보정기

code/<subdir>/ 로 옮겨진 스크립트들은 원래 ver1/<folder>/ 에 있었기 때문에
'../data', '../RAW' 같은 상대경로나 SCRIPT_DIR=dirname(__file__) 기반 경로가
한 단계씩 어긋난다. 이 스크립트가 두 가지를 자동 보정한다.

  (A) literal 상대경로용 → 파일 상단에 chdir 부트스트랩 삽입
      (cwd 를 code/ 로 고정 → '../data' = ver1/data 로 원복)
  (B) SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
      → 한 단계 위(code/)를 가리키도록 교체 → SCRIPT_DIR/../data = ver1/data

멱등(idempotent): 이미 처리된 파일(__pathfix__ 마커)은 건너뛴다.
"""
import os
import re

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MARKER = "# __pathfix__"

BOOTSTRAP = (
    f"{MARKER} (code/<sub>/ 로 옮긴 스크립트가 원래 ver1/ 기준 경로를 찾도록 보정)\n"
    "import os as _os\n"
    "__file__ = _os.path.abspath(__file__)  # cwd 변경 전에 절대경로로 고정\n"
    "_os.chdir(_os.path.dirname(_os.path.dirname(__file__)))  # cwd = ver1/code/\n"
    "del _os\n"
)

SCRIPT_DIR_OLD = "SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))"
SCRIPT_DIR_NEW = (
    "SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
    f"  {MARKER} code/ 기준 (= 원래 스크립트 폴더와 같은 깊이)"
)


def find_insert_index(lines):
    """셔뱅/코딩/상단주석/상단 docstring 다음, 첫 코드 직전 줄 index."""
    i = 0
    n = len(lines)
    # 셔뱅
    if i < n and lines[i].startswith("#!"):
        i += 1
    # 코딩선언 및 상단 연속 주석/공백
    while i < n and (lines[i].lstrip().startswith("#") or lines[i].strip() == ""):
        i += 1
    # 상단 모듈 docstring
    if i < n:
        s = lines[i].lstrip()
        for q in ('"""', "'''"):
            if s.startswith(q):
                # 한 줄 docstring?
                if s.count(q) >= 2 and len(s.strip()) > len(q):
                    i += 1
                else:
                    i += 1
                    while i < n and q not in lines[i]:
                        i += 1
                    if i < n:
                        i += 1
                break
    return i


def process(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if MARKER in text:
        return "skip(이미처리)"
    lines = text.splitlines(keepends=True)

    changed = False
    # (B) SCRIPT_DIR 교체
    if SCRIPT_DIR_OLD in text:
        text = text.replace(SCRIPT_DIR_OLD, SCRIPT_DIR_NEW)
        lines = text.splitlines(keepends=True)
        changed = True

    # (A) chdir 부트스트랩 삽입
    idx = find_insert_index(lines)
    block = BOOTSTRAP if (lines and lines[idx - 1].endswith("\n")) else "\n" + BOOTSTRAP
    block = block + "\n"
    lines.insert(idx, block)
    changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return "fixed"
    return "nochange"


def main():
    results = {}
    for root, _, files in os.walk(CODE_DIR):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            if fn.startswith("_"):  # _common.py, _apply_pathfix.py 제외
                continue
            p = os.path.join(root, fn)
            results[os.path.relpath(p, CODE_DIR)] = process(p)
    for k in sorted(results):
        print(f"{results[k]:14s} {k}")
    print(f"\n총 {len(results)}개 처리")


if __name__ == "__main__":
    main()
