from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CJK_TEXT = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def test_all_tracked_text_is_english_only() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    violations: list[str] = []
    for encoded_path in tracked:
        if not encoded_path:
            continue
        relative_path = encoded_path.decode("utf-8")
        data = (ROOT / relative_path).read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _CJK_TEXT.search(line):
                violations.append(f"{relative_path}:{line_number}")
    assert not violations, "Non-English CJK text found in:\n" + "\n".join(violations)
