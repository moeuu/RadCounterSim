"""Dependency-light HTML run report rendering."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _rows(mapping: dict[str, Any]) -> str:
    return "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in sorted(mapping.items())
    )


def render_run_report(run_directory: str | Path, output_path: str | Path) -> Path:
    """Render manifest and metrics into a self-contained report."""

    root = Path(run_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    artifacts = sorted(path.name for path in root.iterdir() if path.is_file())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>RadCounterSim run report</title>
<style>body{{font-family:Georgia,serif;max-width:960px;margin:3rem auto;color:#17211b}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #b8c2ba;
padding:.45rem;text-align:left}}th{{background:#edf2ed;width:32%}}code{{font-family:monospace}}
h1,h2{{color:#184b35}}</style></head><body><h1>RadCounterSim run report</h1>
<h2>Metrics</h2><table>{_rows(metrics)}</table>
<h2>Manifest</h2><table>{_rows(manifest)}</table>
<h2>Artifacts</h2><ul>{"".join(f"<li>{html.escape(name)}</li>" for name in artifacts)}</ul>
</body></html>"""
    output = Path(output_path)
    output.write_text(document, encoding="utf-8")
    return output
