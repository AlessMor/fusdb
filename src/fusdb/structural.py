"""Structural decomposition helpers for relation-variable incidence graphs."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def structural_decomposition(
    *,
    relations: list[object],
    variables: list[str],
    matrix: coo_matrix,
) -> dict[str, object]:
    """Return Dulmage-Mendelsohn partitions and block decomposition.

    Args:
        relations: Row relation nodes.
        variables: Column variable names.
        matrix: Row x column sparse incidence matrix.

    Returns:
        Structural decomposition payload with partitions and blocks.
    """
    # Keep empty cases explicit and deterministic.
    n_rows, n_cols = matrix.shape
    if n_rows == 0 or n_cols == 0:
        return {
            "relations": relations,
            "variables": variables,
            "matrix_shape": (n_rows, n_cols),
            "row_partitions": {"under": [], "well": [], "over": []},
            "col_partitions": {"under": [], "well": [], "over": []},
            "blocks": [],
        }

    # Build one bipartite graph view and one sparse matching for DM.
    graph = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(matrix)
    top_nodes = list(range(n_rows))
    top_node_set = set(top_nodes)
    bottom_nodes = [node for node in graph if node not in top_node_set]
    row_node_to_idx = {node: idx for idx, node in enumerate(top_nodes)}
    col_node_to_idx = {node: idx for idx, node in enumerate(bottom_nodes)}
    matching: dict[object, object] = {}
    try:
        row_to_col = maximum_bipartite_matching(matrix.tocsr(), perm_type="column")
    except Exception:
        row_to_col = np.full(n_rows, -1, dtype=int)
    for row_idx, col_idx in enumerate(np.asarray(row_to_col, dtype=int).tolist()):
        if col_idx < 0 or row_idx >= len(top_nodes) or col_idx >= len(bottom_nodes):
            continue
        row_node = top_nodes[row_idx]
        col_node = bottom_nodes[int(col_idx)]
        matching[row_node] = col_node
        matching[col_node] = row_node
    dm_rows, dm_cols = dulmage_mendelsohn(
        graph,
        top_nodes=top_nodes,
        matching=matching if matching else None,
    )

    # Convert DM node partitions to row/column index partitions.
    t_unmatched, t_reachable, t_matched, t_other = dm_rows
    b_unmatched, b_reachable, b_matched, b_other = dm_cols
    row_under = sorted(
        {row_node_to_idx[node] for node in t_matched if node in row_node_to_idx}
    )
    row_over = sorted(
        {
            row_node_to_idx[node]
            for node in itertools.chain(t_unmatched, t_reachable)
            if node in row_node_to_idx
        }
    )
    row_well = sorted(
        {row_node_to_idx[node] for node in t_other if node in row_node_to_idx}
    )
    col_under = sorted(
        {
            col_node_to_idx[node]
            for node in itertools.chain(b_unmatched, b_reachable)
            if node in col_node_to_idx
        }
    )
    col_over = sorted(
        {col_node_to_idx[node] for node in b_matched if node in col_node_to_idx}
    )
    col_well = sorted(
        {col_node_to_idx[node] for node in b_other if node in col_node_to_idx}
    )

    # Triangularize only the square well-constrained core.
    blocks: list[tuple[list[int], list[int]]] = []
    if row_well and col_well and len(row_well) == len(col_well):
        sub = matrix.tocsr()[row_well, :][:, col_well].tocoo()
        try:
            n_sub_rows, n_sub_cols = sub.shape
            sub_matching: dict[int, int] = {}
            if n_sub_rows and n_sub_cols:
                row_to_col = maximum_bipartite_matching(sub.tocsr(), perm_type="column")
                for row_idx, col_idx in enumerate(np.asarray(row_to_col, dtype=int).tolist()):
                    if col_idx < 0:
                        continue
                    col_node = n_sub_rows + int(col_idx)
                    sub_matching[int(row_idx)] = col_node
                    sub_matching[col_node] = int(row_idx)
            row_parts, col_parts = block_triangularize(
                sub,
                matching=sub_matching if sub_matching else None,
            )
            for row_part, col_part in zip(row_parts, col_parts, strict=True):
                block_rows = [row_well[idx] for idx in row_part]
                block_cols = [col_well[idx] for idx in col_part]
                blocks.append((block_rows, block_cols))
        except Exception:
            blocks = [(list(row_well), list(col_well))]

    return {
        "relations": relations,
        "variables": variables,
        "matrix_shape": (n_rows, n_cols),
        "row_partitions": {
            "under": row_under,
            "well": row_well,
            "over": row_over,
        },
        "col_partitions": {
            "under": col_under,
            "well": col_well,
            "over": col_over,
        },
        "blocks": blocks,
    }
