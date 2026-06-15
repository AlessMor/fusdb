"""Render matplotlib figures to self-contained HTML for embedding in docs.

The fusdb docs site embeds figures through ``<iframe>`` elements (see
``docs/scripts/build_docs_assets.py``); :func:`figure_to_html` turns any figure
into a standalone HTML document suitable as the iframe source.
"""

from __future__ import annotations

import base64
import io

from matplotlib.figure import Figure

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  .fusdb-figure {{ width: 100%; padding: 8px; box-sizing: border-box; }}
  .fusdb-figure svg, .fusdb-figure img {{ width: 100%; height: auto; display: block; }}
</style>
</head>
<body>
<div class="fusdb-figure">{media}</div>
</body>
</html>
"""


def figure_to_html(figure: Figure, *, fmt: str = "svg", title: str = "", dpi: int = 150) -> str:
    """Return a self-contained HTML document embedding ``figure``.

    Args:
        figure: Matplotlib figure to embed.
        fmt: ``"svg"`` for a crisp, scalable inline vector (best for line plots)
            or ``"png"`` for a base64 ``<img>`` (best for dense scatter/graphs).
        title: Document ``<title>`` and image alt text.
        dpi: Raster resolution when ``fmt="png"``.

    Returns:
        A complete HTML document string, usable directly as an ``<iframe>`` source.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format=fmt, bbox_inches="tight", dpi=dpi)
    payload = buffer.getvalue()

    if fmt == "svg":
        svg = payload.decode("utf-8")
        media = svg[svg.index("<svg"):]  # drop the XML prolog/doctype so it inlines cleanly
    elif fmt == "png":
        encoded = base64.b64encode(payload).decode("ascii")
        media = f'<img src="data:image/png;base64,{encoded}" alt="{title}">'
    else:
        raise ValueError(f"Unsupported fmt {fmt!r}; use 'svg' or 'png'.")

    return _TEMPLATE.format(title=title or "figure", media=media)
