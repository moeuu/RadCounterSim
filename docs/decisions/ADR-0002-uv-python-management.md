# ADR-0002: uv-only Python management

Status: Accepted

The project uses `pyproject.toml`, `uv.lock`, dependency groups, `uv sync`, and
`uv run`. System Python and ad-hoc `pip install` commands are unsupported.
