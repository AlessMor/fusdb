from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HEAD = Path(os.environ["GITHUB_WORKSPACE"])
MAIN = Path("/tmp/fusdb-main")
MARKER = "FUSDB_DIAG_JSON="

CODE = r'''
from pathlib import Path
import json
import time
from fusdb import Reactor

arc = Path("reactors/ARC_V0/reactor.yaml")
r = Reactor.from_yaml(arc)
candidate = r._clone_for_regime("i_mode", include_guards=False)

t0 = time.perf_counter()
s = candidate.relation_system()
build_s = time.perf_counter() - t0

t0 = time.perf_counter()
s.compile()
compile_s = time.perf_counter() - t0
x0, _lo, _hi = s.pack()
vals = s.unpack(x0)
layout = s.residual_layout(vals, include_movement=True)
try:
    sp = s.build_jac_sparsity(layout)
    nnz = None if sp is None else int(sp.nnz)
except Exception:
    nnz = None

struct = {
    "build_s": build_s,
    "compile_s": compile_s,
    "candidate_relations": len(s.candidate_primary_relations),
    "active_relations": len(s.relations),
    "providers": len(getattr(s, "_provider_plan", ())),
    "completion_passes": int(getattr(s, "_completion_passes", -1)),
    "dim": int(x0.size),
    "residual_size": int(layout["size"]),
    "jac_nnz": nnz,
}

t0 = time.perf_counter()
direct = s.reconcile()
struct["direct_s"] = time.perf_counter() - t0
solver = direct.get("solver") or {}
struct["direct_ok"] = bool(direct.get("success"))
struct["residual_calls"] = solver.get("residual_calls")
struct["residual_mean_ms"] = solver.get("residual_eval_mean_ms")
struct["nfev"] = solver.get("nfev")
struct["stage_history"] = [
    {
        "stage": x.get("stage"),
        "nfev": x.get("nfev"),
        "elapsed_s": x.get("elapsed_s"),
        "jac_mode": x.get("jac_mode"),
        "residual_size": x.get("residual_size"),
        "verified": x.get("verified"),
    }
    for x in (solver.get("stage_history") or [])
]
print("FUSDB_DIAG_JSON=" + json.dumps(struct, separators=(",", ":")))
'''


def run(label: str, root: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    p = subprocess.run(
        [sys.executable, "-c", CODE], cwd=root, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if p.returncode:
        print(p.stdout)
        raise SystemExit(f"{label} diagnostic failed")
    lines = [line for line in p.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        print(p.stdout)
        raise SystemExit(f"{label} diagnostic marker count {len(lines)}")
    data = json.loads(lines[0][len(MARKER):])
    print(label, json.dumps(data, indent=2, sort_keys=True))
    return data


def post(context: str, description: str) -> None:
    payload = json.dumps({"state": "success", "context": context, "description": description[:140]})
    url = f"https://api.github.com/repos/{os.environ['REPO']}/statuses/{os.environ['HEAD_SHA']}"
    subprocess.run([
        "curl", "-sS", "-L", "-X", "POST",
        "-H", "Accept: application/vnd.github+json",
        "-H", f"Authorization: Bearer {os.environ['GH_TOKEN']}",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        url, "-d", payload,
    ], check=True)


def main() -> None:
    main_data = run("main", MAIN)
    head_data = run("head", HEAD)
    post(
        "fusdb/diag/reconcile-structure",
        "main/head rel=%s/%s dim=%s/%s rows=%s/%s nnz=%s/%s providers=%s/%s"
        % (
            main_data["active_relations"], head_data["active_relations"],
            main_data["dim"], head_data["dim"],
            main_data["residual_size"], head_data["residual_size"],
            main_data["jac_nnz"], head_data["jac_nnz"],
            main_data["providers"], head_data["providers"],
        ),
    )
    post(
        "fusdb/diag/reconcile-runtime",
        "main/head solve=%.2f/%.2fs calls=%s/%s mean=%.2f/%.2fms nfev=%s/%s stages=%s/%s"
        % (
            main_data["direct_s"], head_data["direct_s"],
            main_data["residual_calls"], head_data["residual_calls"],
            main_data["residual_mean_ms"], head_data["residual_mean_ms"],
            main_data["nfev"], head_data["nfev"],
            len(main_data["stage_history"]), len(head_data["stage_history"]),
        ),
    )
    post(
        "fusdb/diag/reconcile-stages",
        "main=" + json.dumps(main_data["stage_history"], separators=(",", ":"))[:55]
        + " head=" + json.dumps(head_data["stage_history"], separators=(",", ":"))[:55],
    )


if __name__ == "__main__":
    main()
