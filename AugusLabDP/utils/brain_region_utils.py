"""Utilities for brain region mapping and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_STRUCTURE_GRAPH_PATH = Path(__file__).with_name("structure_graph_allen_ccf.json")
_REGION_INFO_PATH = Path(__file__).with_name("region_info.csv") # IBL region csv file


class StructureTree:
    """Small Allen structure graph reader backed by the bundled JSON file."""

    def __init__(self, structure_graph_path: str | Path = _STRUCTURE_GRAPH_PATH) -> None:
        with Path(structure_graph_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        root = data["msg"][0] if isinstance(data, dict) and "msg" in data else data

        self._by_id: Dict[int, Dict[str, Any]] = {}
        self._by_acronym: Dict[str, Dict[str, Any]] = {}
        self._add_node(root, [])

    def _add_node(self, node: Dict[str, Any], parent_path: List[int]) -> None:
        node = dict(node)
        children = node.pop("children", [])
        node["structure_id_path"] = [*parent_path, node["id"]]
        self._by_id[node["id"]] = node
        self._by_acronym[node["acronym"]] = node
        for child in children:
            self._add_node(child, node["structure_id_path"])

    def get_structures_by_acronym(self, acronyms: List[str]) -> List[Dict[str, Any]]:
        return [self._by_acronym[acronym] for acronym in acronyms if acronym in self._by_acronym]

    def get_structures_by_id(self, structure_ids: List[int]) -> List[Dict[str, Any]]:
        return [self._by_id[structure_id] for structure_id in structure_ids if structure_id in self._by_id]

    def get_colors_by_acronym(self, acronyms: List[str]) -> List[str]:
        return ['#' + self._by_acronym[acronym]["color_hex_triplet"] for acronym in acronyms if acronym in self._by_acronym]

tree = StructureTree()

def _structure_by_acronym(acronym: str) -> Dict[str, Any]:
    structures = tree.get_structures_by_acronym([acronym])
    if not structures:
        raise KeyError(f"Unknown Allen brain region acronym: {acronym}")
    return structures[0]


def _structure_by_id(structure_id: int) -> Dict[str, Any]:
    structures = tree.get_structures_by_id([structure_id])
    if not structures:
        raise KeyError(f"Unknown Allen brain region id: {structure_id}")
    return structures[0]



# meta_region_color_map: Dict[str, str] = {
#     "Cerebrum": "#B0F0FF",
#     "Cerebral cortex": "#B0FFB8",
#     "Cerebral nuclei": "#98D6F9",
#     "Brain stem": "#FF7080",
#     "Interbrain": "#FF7080",
#     "Midbrain": "#FF64FF",
#     "Hindbrain": "#FF9B88",
#     "Cerebellum": "#F0F080",
#     "Cerebellar cortex": "#F0F080",
#     "Cerebellar nuclei": "#F0F080",
#     "CH": "#B0F0FF",
#     "CTX": "#B0FFB8",
#     "CNU": "#98D6F9",
#     "BS": "#FF7080",
#     "IB": "#FF7080",
#     "MB": "#FF64FF",
#     "HB": "#FF9B88",
#     "CB": "#F0F080",
#     "CBX": "#F0F080",
#     "CBN": "#F0F080",
# }



TARGET_REGION_LIST: List[str] = ["TH", "HY", "HPF", "BS", "CTX", "CNU"]
TARGET_REGION_LIST = sorted(
    TARGET_REGION_LIST,
    key=lambda x: len(_structure_by_acronym(x)["structure_id_path"]),
    reverse=True
)
# No need to sort by id path length; keep user ordering so earlier regions "catch" first.

def get_meta_region_by_target_list(
    cluster_region_all: np.ndarray,
    target_region_list: Optional[List[str]] = None,
) -> np.ndarray:
    """Map brain region acronyms to the first matching target ancestor in target_region_list (priority: earlier item)."""
    target_region_list = target_region_list or TARGET_REGION_LIST
    # Precompute structure id for each target region in user order
    target_ids_in_order = [
        (_structure_by_acronym(tr), _structure_by_acronym(tr)["id"], tr)
        for tr in target_region_list
    ]
    meta_region_all = []
    for region in cluster_region_all:
        try:
            path = _structure_by_acronym(region)["structure_id_path"]
        except Exception:
            meta_region_all.append("other")
            continue
        # Use the first match according to given order
        matched_region = "other"
        for _, region_id, tr in target_ids_in_order:
            if region_id in path:
                matched_region = tr
                break
        meta_region_all.append(matched_region)
    return np.array(meta_region_all)


def get_meta_region_coarse(cluster_region_all: np.ndarray) -> np.ndarray:
    """Map brain region acronyms to coarse CCF hierarchy labels."""
    meta_region_all = []
    depth_rules = [
        (("Isocortex", "OLF"), 6),
        (("CA",), 8),
        (("DG", "FC", "IG"), 7),
        (("CTXsp",), 5),
        (("RHP",), 6),
        (("CNU",), 3),
        (("HY", "TH"), 5),
        (("MB", "HB"), 4),
        (("CB",), 3),
    ]

    for region in cluster_region_all:
        if region in {"outside_brain", None, ""}:
            meta_region_all.append("outside_brain")
            continue

        try:
            path_ids = _structure_by_acronym(region)["structure_id_path"][1:]
            structure_names = [_structure_by_id(region_id)["acronym"] for region_id in path_ids]
        except Exception:
            meta_region_all.append("outside_brain")
            continue

        depth = next(
            (depth for acronyms, depth in depth_rules if any(a in structure_names for a in acronyms)),
            1,
        )
        meta_region_all.append(
            structure_names[depth - 1]
            if len(structure_names) >= depth
            else structure_names[-1]
            if structure_names
            else "outside_brain"
        )
    return np.array(meta_region_all)


def get_meta_region_IBL(
    cluster_region_all: np.ndarray,
    region_info_path: str | Path = _REGION_INFO_PATH,
) -> np.ndarray:
    """Map brain region acronyms to IBL/Beryl-style meta-region labels."""
    region_info = pd.read_csv(region_info_path)
    beryl_regions = set(region_info["Beryl"].dropna())
    bottomline_meta_region = {
        "Isocortex",
        "OLF",
        "HPF",
        "CNU",
        "CTXsp",
        "TH",
        "HY",
        "MB",
        "HB",
        "fiber tracts",
        "VS",
    }

    meta_region_all = []
    for region in cluster_region_all:
        if region == "outside_brain":
            meta_region_all.append("outside_brain")
            continue

        region_id_path = _structure_by_acronym(region)["structure_id_path"]
        for region_id in reversed(region_id_path):
            region_name = _structure_by_id(region_id)["acronym"]
            if region_name in beryl_regions or region_name in bottomline_meta_region:
                meta_region_all.append(region_name)
                break
        else:
            meta_region_all.append("other")
    return np.array(meta_region_all)

def mark_region_cluster(arr: np.ndarray) -> List[Tuple[Any, int, int]]:
    """Return contiguous repeated regions as ``(value, start, end)`` tuples."""
    if len(arr) == 0:
        return []

    repeats: List[Tuple[Any, int, int]] = []
    start = end = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            if end > start:
                repeats.append((arr[start], start, end))
            start = end = i
        else:
            end = i

    if end > start:
        repeats.append((arr[start], start, end))
    return repeats

def plot_region_mark(
    cluster_region: np.ndarray,
    ax: Optional[Any] = None,
    orientation: str = "v",
    reversed: Optional[bool] = None,
    show_figure: bool = True,
    fill_ratio: float = 1,
    get_meta_region_func: Callable[[np.ndarray], np.ndarray] = get_meta_region_IBL,
) -> Optional[Any]:
    """Plot brain region boundaries as colored spans on an axis."""
    orientation = {"vertical": "v", "v": "v", "horizontal": "h", "h": "h"}.get(orientation)
    if orientation is None:
        raise ValueError("Orientation can only be v/vertical/h/horizontal")

    if reversed if reversed is not None else orientation == "v":
        cluster_region = cluster_region[::-1]

    region_rep = mark_region_cluster(cluster_region)
    meta_region_list = get_meta_region_func(cluster_region)
    meta_region_rep = mark_region_cluster(meta_region_list)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2) if orientation == "h" else (2, 10))

    ticklabels = []
    tickpos = []

    meta_region_colors = tree.get_colors_by_acronym(np.unique(meta_region_list))
    color_map = {meta_region: color for meta_region, color in zip(np.unique(meta_region_list), meta_region_colors)}

    for region, start, end in meta_region_rep:
        color = color_map.get(region, "white")
        if orientation == "h":
            ax.axvspan(start - 0.5, end + 0.5, ymin=1 - fill_ratio, alpha=0.2, color=color)
        else:
            ax.axhspan(start - 0.5, end + 0.5, xmin=1 - fill_ratio, alpha=0.2, color=color)

    for region, start, end in region_rep:
        if end - start < 10:
            continue
        tickpos.append((start + end) / 2)
        ticklabels.append(region)
        if orientation == "h":
            ax.axvline(end, 1 - fill_ratio, 1, color="black", linewidth=0.5)
        else:
            ax.axhline(end, 1 - fill_ratio, 1, color="black", linewidth=0.5)

    if orientation == "h":
        ax.axvline(0, 1 - fill_ratio, 1, color="black")
        ax.set_xticks(tickpos, ticklabels, rotation=90)
        ax.set_yticks([], [])
    else:
        ax.axhline(0, 1 - fill_ratio, 1, color="black")
        ax.set_yticks(tickpos, ticklabels)
        ax.set_xticks([], [])

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    if show_figure:
        plt.tight_layout()
        plt.show()
        return None
    return ax

