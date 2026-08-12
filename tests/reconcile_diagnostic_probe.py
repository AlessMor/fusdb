from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HEAD = Path(os.environ["GITHUB_WORKSPACE"])
MAIN = Path("/tmp/fusdb-main")
ROOTS = {"main": MAIN, "head": HEAD}
MARKER = "FUSDB_RECON_DIAG="

PROBE = r'''
import json
import time
from pathlib import Path
from fusdb import Reactor

reactor = Reactor.from_yaml(Path("reactors/ARC_V0/reactor.yaml"))
clone = reactor._clone_for_regime("i_mode", include_guards=False)
system = clone.relation_system()
system.compile()
relation_names = [rel.name for rel in system.relations]
residual_names = [rel.name for rel in system._enforced_residual_relations]
provider_names = sorted({rel.name for rel in system.derived_provider_by_output.values()} | {rel.name for rel in system.default_provider_by_output.values()})
packed_dim = int(system.packed_dim)
start = time.perf_counter()
result = system.run("reconcile")
elapsed = time.perf_counter() - start
solver = result.get("solver") or {}
print("FUSDB_RECON_DIAG=" + json.dumps({
    "elapsed": elapsed,
    "success": bool(result.get("success")),
    "verified": bool(result.get("verified", result.get("success"))),
    "relations": relation_names,
    "residual_relations": residual_names,
    "providers": provider_names,
    "packed_dim": packed_dim,
    "residual_size": int(solver.get("residual_size", -1)),
    "residual_calls": int(solver.get("residual_calls", -1)),
    "residual_eval_mean_ms": float(solver.get("residual_eval_mean_ms", float("nan"))),
    "stages": solver.get("stage_history") or [],
    "beyond": [item.get("variable", item.get("name", "?")) if isinstance(item, dict) else str(item) for item in (result.get("inputs_beyond_tolerance") or [])],
}, sort_keys=True))
'''


def run(label: str) -> dict:
    root = ROOTS[label]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        raise SystemExit(f"{label} reconcile diagnostic failed")
    payloads = [
        json.loads(line[len(MARKER):])
        for line in completed.stdout.splitlines()
        if line.startswith(MARKER)
    ]
    if len(payloads) != 1:
        print(completed.stdout)
        raise SystemExit(f"{label} diagnostic emitted {len(payloads)} markers")
    return payloads[0]


def compact_stage(stage: dict) -> str:
    label = stage.get("stage")
    nfev = stage.get("nfev")
    elapsed = float(stage.get("elapsed_s", 0.0))
    failed = stage.get("failed_relations")
    beyond = stage.get("inputs_beyond_tolerance", "-")
    return f"{label}:n{nfev}/t{elapsed:.1f}/f{failed}/b{beyond}"


def publish(context: str, description: str, state: str = "success") -> None:
    payload = json.dumps({
        "state": state,
        "context": context,
        "description": description[:140],
    })
    url = f"https://api.github.com/repos/{os.environ['REPO']}/statuses/{os.environ['HEAD_SHA']}"
    subprocess.run(
        [
            "curl", "-sS", "-L", "-X", "POST",
            "-H", "Accept: application/vnd.github+json",
            "-H", f"Authorization: Bearer {os.environ['GH_TOKEN']}",
            "-H", "X-GitHub-Api-Version: 2022-11-28",
            url, "-d", payload,
        ],
        check=True,
    )


def main() -> None:
    main_result = run("main")
    head_result = run("head")
    main_rel = set(main_result["relations"])
    head_rel = set(head_result["relations"])
    main_res = set(main_result["residual_relations"])
    head_res = set(head_result["residual_relations"])
    main_prov = set(main_result["providers"])
    head_prov = set(head_result["providers"])

    print(json.dumps({"main": main_result, "head": head_result}, indent=2, sort_keys=True))
    publish(
        "fusdb/diag/fixed-runtime",
        f"main/head {main_result['elapsed']:.2f}/{head_result['elapsed']:.2f}s calls {main_result['residual_calls']}/{head_result['residual_calls']} mean {main_result['residual_eval_mean_ms']:.2f}/{head_result['residual_eval_mean_ms']:.2f}ms",
    )
    publish(
        "fusdb/diag/fixed-structure",
        f"dim {main_result['packed_dim']}/{head_result['packed_dim']} rows {main_result['residual_size']}/{head_result['residual_size']} rel +{len(head_rel-main_rel)} -{len(main_rel-head_rel)} res +{len(head_res-main_res)} -{len(main_res-head_res)} prov +{len(head_prov-main_prov)} -{len(main_prov-head_prov)}",
    )
    publish(
        "fusdb/diag/fixed-added-relations",
        "add=" + ",".join(sorted(head_rel - main_rel)) + " | del=" + ",".join(sorted(main_rel - head_rel)),
    )
    publish(
        "fusdb/diag/fixed-added-residuals",
        "add=" + ",".join(sorted(head_res - main_res)) + " | del=" + ",".join(sorted(main_res - head_res)),
    )
    publish(
        "fusdb/diag/fixed-added-providers",
        "add=" + ",".join(sorted(head_prov - main_prov)) + " | del=" + ",".join(sorted(main_prov - head_prov)),
    )
    publish(
        "fusdb/diag/fixed-stages-main",
        " ".join(compact_stage(stage) for stage in main_result["stages"]),
    )
    publish(
        "fusdb/diag/fixed-stages-head",
        " ".join(compact_stage(stage) for stage in head_result["stages"]),
    )
    publish(
        "fusdb/diag/fixed-beyond",
        "main=" + ",".join(main_result["beyond"]) + " | head=" + ",".join(head_result["beyond"]),
    )


if __name__ == "__main__":
    main()
