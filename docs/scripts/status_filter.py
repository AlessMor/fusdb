from __future__ import annotations

from pathlib import Path
import re

from mkdocs.structure.files import Files

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_STATUS_RE = re.compile(r"^\s*status\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _page_status(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return None

    status_match = _STATUS_RE.search(match.group(1))
    if status_match is None:
        return None
    return status_match.group(1).strip().strip("'\"").lower()


def on_files(files, *, config):
    filtered = Files([])
    docs_dir = Path(config["docs_dir"]).resolve()

    for file in files:
        if not file.src_uri.endswith(".md"):
            filtered.append(file)
            continue

        abs_src_path = getattr(file, "abs_src_path", None)
        path = Path(abs_src_path) if abs_src_path else docs_dir / file.src_uri
        status = _page_status(path)
        if status == "draft":
            continue
        filtered.append(file)

    return filtered
