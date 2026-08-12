from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HEAD = Path(os.environ["GITHUB_WORKSPACE"])
MAIN = Path("/tmp/fusdb-main")
ROOTS = {"main": MAIN, "head": HEAD}
MARKER = "FUSDB_BENCH_SECONDS="
ORDER = ("main", "head", "head", "main", "main", "head")

SCRIPTS = {
    "reconcile": '''
from pathlib import Path
import time
from fusdb import Reactor
r = Reactor.from_yaml(Path("reactors/ARC_V0/reactor.yaml"))
t0 = time.perf_counter()
result = r.reconcile()
elapsed = time.perf_counter() - t0
assert result["success"], result.get("errors")
print("FUSDB_BENCH_SECONDS=" + repr(elapsed))
''',
    "popcon": '''
from pathlib import Path
import time
from fusdb import Reactor
r = Reactor.from_yaml(Path("tests/cfspopcon_SPARC/reactor.yaml"))
t0 = time.perf_counter()
result = r.popcon(
    x={"variable": "average_electron_density", "values": [2.5e20, 3.0e20]},
    y={"variable": "average_electron_temp", "values": [9.0, 12.0]},
    outputs=("P_fus", "P_aux"),
)
elapsed = time.perf_counter() - t0
assert result["success"], result.get("errors")
assert result["popcon"]["success"].all(), result["popcon"]["failures"]
print("FUSDB_BENCH_SECONDS=" + repr(elapsed))
''',
}


def measured_run(label: str, workload: str, code: str) -> float:
    root = ROOTS[label]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        raise SystemExit(f"{label} {workload} benchmark failed")
    timings = [
        float(line[len(MARKER):])
        for line in completed.stdout.splitlines()
        if line.startswith(MARKER)
    ]
    if len(timings) != 1:
        print(completed.stdout)
        raise SystemExit(f"{label} {workload} emitted {len(timings)} timing markers")
    return timings[0]


def publish_status(workload: str, values: dict[str, float]) -> bool:
    ratio = values["ratio"]
    state = "success" if ratio <= 1.10 else "failure"
    payload = json.dumps(
        {
            "state": state,
            "context": f"fusdb/perf/{workload}",
            "description": (
                f"main {values['main']:.2f}s, head {values['head']:.2f}s, "
                f"ratio {ratio:.3f} (limit 1.10)"
            ),
        }
    )
    url = (
        f"https://api.github.com/repos/{os.environ['REPO']}"
        f"/statuses/{os.environ['HEAD_SHA']}"
    )
    completed = subprocess.run(
        [
            "curl",
            "-sS",
            "-L",
            "-X",
            "POST",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            f"Authorization: Bearer {os.environ['GH_TOKEN']}",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
            url,
            "-d",
            payload,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"failed to publish {workload} performance status")
    return state == "failure"


def main() -> None:
    results: dict[str, dict[str, float]] = {}
    for workload, code in SCRIPTS.items():
        samples = {"main": [], "head": []}
        for label in ORDER:
            elapsed = measured_run(label, workload, code)
            samples[label].append(elapsed)
            print(f"{workload} {label}: {elapsed:.6f}s")
        baseline = statistics.median(samples["main"])
        candidate = statistics.median(samples["head"])
        results[workload] = {
            "main": baseline,
            "head": candidate,
            "ratio": candidate / baseline,
        }

    print(json.dumps(results, indent=2, sort_keys=True))
    failed = any(publish_status(workload, values) for workload, values in results.items())
    if failed:
        raise SystemExit("Performance regression exceeded 10% threshold")


if __name__ == "__main__":
    main()
