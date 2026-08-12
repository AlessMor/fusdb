from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from fusdb import Reactor


result = Reactor.from_yaml(Path("reactors/STELLARIS/reactor.yaml")).run("reconcile")
values = result.get("values") or {}
bad = result.get("inputs_beyond_tolerance") or []
bad_names = [
    str(item.get("variable", item.get("name", "?"))) if isinstance(item, dict) else str(item)
    for item in bad
]
description = (
    f"ok={bool(result.get('success'))} "
    f"tp={values.get('tau_p')!r} te={values.get('tau_E')!r} "
    f"Te={values.get('T_e_avg')!r} ne={values.get('n_e_avg')!r} "
    f"bad={len(bad)}:{','.join(bad_names)}"
)[:140]
payload = json.dumps(
    {
        "state": "success",
        "context": "fusdb/diag/stellaris-values",
        "description": description,
    }
).encode()
request = urllib.request.Request(
    f"https://api.github.com/repos/{os.environ['REPO']}/statuses/{os.environ['SHA']}",
    data=payload,
    method="POST",
    headers={
        "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    },
)
with urllib.request.urlopen(request) as response:
    if response.status >= 300:
        raise SystemExit(f"status publish failed: {response.status}")
print(description)
