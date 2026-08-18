from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("src/fusdb")
OUT = Path(".architecture-duplication-report.txt")


def source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def normalized_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    # Compare implementation rather than names/signatures/docstrings.  Local
    # identifier names are intentionally retained: an exact body match is a
    # high-confidence duplication signal, not a fuzzy similarity guess.
    module = ast.Module(body=body_without_docstring(node), type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def function_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        lines = text.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = body_without_docstring(node)
                records.append(
                    {
                        "path": path,
                        "qualname": node.name,
                        "node": node,
                        "body": body,
                        "normalized": normalized_body(node),
                        "line_count": (node.end_lineno or node.lineno) - node.lineno + 1,
                        "statement_count": len(body),
                        "snippet": "\n".join(lines[node.lineno - 1 : node.end_lineno]),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        body = body_without_docstring(child)
                        records.append(
                            {
                                "path": path,
                                "qualname": f"{node.name}.{child.name}",
                                "node": child,
                                "body": body,
                                "normalized": normalized_body(child),
                                "line_count": (child.end_lineno or child.lineno) - child.lineno + 1,
                                "statement_count": len(body),
                                "snippet": "\n".join(lines[child.lineno - 1 : child.end_lineno]),
                            }
                        )
    return records


def identifier_references() -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                counts[node.id] += 1
    return counts


def main() -> None:
    records = function_records()
    refs = identifier_references()
    lines: list[str] = []

    root_files = sorted(path.name for path in ROOT.glob("*.py"))
    lines += [
        "ARCHITECTURE DUPLICATION REPORT",
        "",
        "ROOT PYTHON FILES",
        *[f"- {name}" for name in root_files],
        "",
    ]

    # Exact implementation duplicates across different files.  Ignore tiny
    # one-statement accessors unless they span enough source to be meaningful.
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        normalized = str(record["normalized"])
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        groups[digest].append(record)
    duplicates = []
    for group in groups.values():
        files = {str(record["path"]) for record in group}
        if len(group) < 2 or len(files) < 2:
            continue
        max_statements = max(int(record["statement_count"]) for record in group)
        max_lines = max(int(record["line_count"]) for record in group)
        if max_statements < 2 and max_lines < 8:
            continue
        duplicates.append(group)

    lines.append("EXACT FUNCTION/METHOD BODY DUPLICATES ACROSS MODULES")
    if not duplicates:
        lines.append("- none above threshold")
    else:
        for group in sorted(duplicates, key=lambda items: (-max(int(r["line_count"]) for r in items), str(items[0]["qualname"]))):
            lines.append(f"- body lines~{max(int(r['line_count']) for r in group)} statements~{max(int(r['statement_count']) for r in group)}")
            for record in group:
                node = record["node"]
                lines.append(f"    {record['path']}:{node.lineno} {record['qualname']}")
    lines.append("")

    lines.append("TRIVIAL TOP-LEVEL WRAPPERS WITH <=1 NAME REFERENCE")
    wrappers = []
    for record in records:
        qualname = str(record["qualname"])
        if "." in qualname:
            continue
        body = record["body"]
        if len(body) != 1:
            continue
        stmt = body[0]
        trivial = isinstance(stmt, ast.Return) and isinstance(stmt.value, (ast.Call, ast.Name, ast.Attribute, ast.Constant, ast.BinOp, ast.UnaryOp))
        if not trivial:
            continue
        # The definition itself is not counted as a Load, so 0/1 is genuinely
        # very small repository-local usage.  Public API exposure still needs a
        # human decision before deletion.
        if refs[qualname] <= 1:
            wrappers.append(record)
    if not wrappers:
        lines.append("- none")
    else:
        for record in sorted(wrappers, key=lambda r: (str(r["path"]), str(r["qualname"]))):
            node = record["node"]
            lines.append(f"- refs={refs[str(record['qualname'])]} {record['path']}:{node.lineno} {record['qualname']}")
    lines.append("")

    lines.append("MOVED-METHOD ALIASES / STALE PATH TEXT")
    needles = ("self = system", "fusdb.seeding", "fusdb.utils", "fusdb.profile_system", "fusdb.profile_sources", "fusdb.batch", "plotting.renderers", "plotting._bokeh", "plotting.tables")
    hits = []
    for path in sorted(ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(needle in line for needle in needles):
                hits.append((path, lineno, line.strip()))
    if not hits:
        lines.append("- none")
    else:
        for path, lineno, line in hits:
            lines.append(f"- {path}:{lineno}: {line}")
    lines.append("")

    lines.append("LARGE SUPPORT MODULES")
    for path in sorted(ROOT.rglob("*.py")):
        if path.name in {"relation.py", "relationsystem.py", "reactor.py"}:
            continue
        n = len(source_lines(path))
        if n >= 500:
            lines.append(f"- {n:5d} lines {path}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
