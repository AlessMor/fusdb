"""Minimal PROCESS MFILE.DAT reader.

Format is fixed-width:  <description padded with _> (<key>)<pad> <value> [OP]
Values are floats, ints, or "quoted strings".  Duplicate keys keep the LAST
occurrence (PROCESS writes scan points sequentially).

The reference files are stored gzipped -- 48 MFILE/OUT/IN files come to 42 MB
raw and 9 MB compressed, and they are checked in.  Both ``.DAT`` and ``.DAT.gz``
are accepted transparently, so callers can name either.
"""

import gzip
import re
import sys
from pathlib import Path

LINE = re.compile(r"^(?P<desc>.*?)_*\s+\((?P<key>[^)]+)\)_*\s+(?P<val>.*?)\s*$")


def open_text(path):
    """Open a PROCESS output file, gzipped or not.

    Accepts a path with or without the ``.gz`` suffix and uses whichever exists,
    so reference paths stay readable regardless of how they are stored.
    """
    path = Path(path)
    if path.suffix == ".gz":
        candidates = [path, path.with_suffix("")]
    else:
        candidates = [path, path.with_suffix(path.suffix + ".gz")]
    for candidate in candidates:
        if candidate.exists():
            opener = gzip.open if candidate.suffix == ".gz" else open
            return opener(candidate, "rt", encoding="utf-8", errors="replace")
    raise FileNotFoundError(f"no such file (with or without .gz): {path}")


def read_mfile(path):
    out = {}
    with open_text(path) as fh:
        for raw in fh:
            if not raw.startswith("#") and "(" in raw:
                m = LINE.match(raw.rstrip("\n"))
                if not m:
                    continue
                key = m.group("key")
                val = m.group("val")
                if val.endswith(" OP"):
                    val = val[:-3].strip()
                if val.startswith('"'):
                    val = val.strip('"').strip()
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        continue
                out[key] = (m.group("desc").rstrip("_").replace("_", " ").strip(), val)
    return out


if __name__ == "__main__":
    data = read_mfile(sys.argv[1])
    print(f"{len(data)} keys")
    for k in sys.argv[2:]:
        print(f"  {k:50s} = {data.get(k, ('<missing>', None))[1]}")
