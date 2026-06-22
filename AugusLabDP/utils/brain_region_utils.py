"""Utilities for brain region mapping and visualization."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import braian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allen Brain Atlas reference space configuration (CCF 2017 / CCFv3)
_STRUCTURE_GRAPH_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
output_dir = os.path.join(str(Path.home()), "allen_reference_atlas")
_ontology_path = Path(output_dir) / "structure_graph_1.json"

if not _ontology_path.exists():
    braian.utils.cache(str(_ontology_path), _STRUCTURE_GRAPH_URL)

_ontology = braian.AllenBrainOntology(str(_ontology_path), version="CCFv3")


class _StructureTree:
    """Thin adapter matching the allensdk structure-tree API used in this module."""

    def __init__(self, ontology: braian.AllenBrainOntology) -> None:
        self._ontology = ontology

    def _structure_id_path(self, acronym: str) -> List[int]:
        ancestors = self._ontology.get_regions_above(acronym)
        path_acronyms = list(reversed(ancestors)) + [acronym]
        return list(self._ontology.acronyms_to_id(path_acronyms))

    def _structure_dict(self, acronym: str) -> Dict[str, Any]:
        structure_id_path = self._structure_id_path(acronym)
        return {
            "id": structure_id_path[-1],
            "acronym": acronym,
            "name": self._ontology.full_name[acronym],
            "structure_id_path": structure_id_path,
        }

    def get_structures_by_acronym(self, acronyms: List[str]) -> List[Dict[str, Any]]:
        return [self._structure_dict(acronym) for acronym in acronyms]

    def get_structures_by_id(self, structure_ids: List[int]) -> List[Dict[str, Any]]:
        acronyms = list(self._ontology.ids_to_acronym(structure_ids))
        return [
            {
                "id": structure_id,
                "acronym": acronym,
                "name": self._ontology.full_name[acronym],
            }
            for structure_id, acronym in zip(structure_ids, acronyms)
        ]


tree = _StructureTree(_ontology)


def mark_region_cluster(arr: np.ndarray) -> List[Tuple[Any, int, int]]:
    """
    Mark the boundaries of consecutive repeated regions in an array.
    
    This function identifies contiguous regions of the same value and returns
    their boundaries as (element, start_index, end_index) tuples.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array containing region labels (can be strings, numbers, etc.).
    
    Returns
    -------
    List[Tuple[Any, int, int]]
        List of tuples, each containing:
        - element: The repeated element value
        - start: Starting index (inclusive)
        - end: Ending index (inclusive)
    """
    n = len(arr)
    if n == 0:
        return []
    repeats = []
    start = 0
    end = 0
    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            if end > start:
                repeats.append((arr[start], start, end))
            start = i
            end = i
        else:
            end = i
    if end > start:
        repeats.append((arr[start], start, end))
    return repeats


def get_meta_region(cluster_region_all: np.ndarray) -> np.ndarray:
    """
    Map brain region acronyms to their meta-regions.
    
    Converts specific brain region acronyms (e.g., 'VISp') to their
    higher-level meta-regions (e.g., 'Cerebrum', 'Cerebellum', 'Brain stem').
    
    Parameters
    ----------
    cluster_region_all : np.ndarray
        Array of brain region acronyms (e.g., ['VISp', 'CA1', 'TH']).
    
    Returns
    -------
    np.ndarray
        Array of meta-region names corresponding to each input region.
        Returns 'outside_brain' for regions not in the brain.
    """
    meta_region_all = []
    for region in cluster_region_all:
        if region == 'outside_brain' or region == 'other':
            meta_region_all.append('outside_brain')
            continue
        path = tree.get_structures_by_acronym([region])[0]['structure_id_path']
        if len(path) <= 2:
            meta_region_id = path[-1]
        elif path[1] == 8:
            if len(path) > 3:
                meta_region_id = path[3]
            else:
                meta_region_id = path[2]
        else:
            meta_region_id = path[1]
        meta_region = tree.get_structures_by_id([meta_region_id])[0]['name']
        # print(region, meta_region)
        meta_region_all.append(meta_region)
    # meta_region_all = [meta_region if meta_region in ['Cerebrum', 'Brain stem', 'Cerebellum'] else 'other' for meta_region in meta_region_all]
        
    meta_region_all = np.array(meta_region_all)
    return meta_region_all

# Color mapping for meta-regions in visualizations
meta_region_color_map: Dict[str, str] = {
    'Cerebrum': '#B0F0FF',
    'Cerebral cortex': '#B0FFB8',
    'Cerebral nuclei': '#98D6F9',
    'Brain stem': '#FF7080',
    'Interbrain': '#FF7080',
    'Midbrain': '#FF64FF',
    'Hindbrain': '#FF9B88',
    'Cerebellum': '#F0F080',
    'Cerebellar cortex': '#F0F080',
    'Cerebellar nuclei': '#F0F080',

    'CH': '#B0F0FF',
    'CTX': '#B0FFB8',
    'CNU': '#98D6F9',
    'BS': '#FF7080',
    'IB': '#FF7080',
    'MB': '#FF64FF',
    'HB': '#FF9B88',
    'CB': '#F0F080',
    'CBX': '#F0F080',
    'CBN': '#F0F080',

}

# Label mapping for experimental conditions
condition_label_map: Dict[str, str] = {
    'iso': 'Isoflurane',
    'syncope': 'Syncope'
}

# Color mapping for experimental conditions
condition_color_map: Dict[str, str] = {
    'iso': '#F89B50',
    'syncope': '#5B84C4',
}
def plot_region_mark(
    cluster_region: np.ndarray,
    ax: Optional[Any] = None,
    orientation: str = 'v',
    reversed: Optional[bool] = None,
    show_figure: bool = True,
    fill_ratio: float = 0.5,
    meta_region_color_map: Optional[Dict[str, str]] = None
) -> Optional[Any]:
    """
    Plot brain region boundaries as colored spans on an axis.
    
    This function visualizes brain regions along a probe track, showing
    both specific regions and their meta-regions with different colors.
    
    Parameters
    ----------
    cluster_region : np.ndarray
        Array of brain region acronyms for each cluster/unit.
    ax : Optional[Any], default None
        Matplotlib axes object. If None, creates a new figure.
    orientation : str, default 'v'
        Orientation of the plot: 'v'/'vertical' or 'h'/'horizontal'.
    reversed : Optional[bool], default None
        Whether to reverse the order of regions. If None, defaults to True for 'v'.
    show_figure : bool, default True
        Whether to display the figure immediately.
    meta_region_color_map : Optional[Dict[str, str]], default None
        Color mapping for meta-regions. If None, uses module default.
    
    Returns
    -------
    Optional[Any]
        Returns the axes object if show_figure is False, otherwise None.
    """
    if meta_region_color_map is None:
        meta_region_color_map = globals()['meta_region_color_map']
    
    if reversed is None:
        reversed = True if orientation == 'v' else False
    if reversed:
        cluster_region = cluster_region[::-1]
    region_rep = mark_region_cluster(cluster_region)

    meta_region = get_meta_region(cluster_region)
    meta_region_rep = mark_region_cluster(meta_region)

    # Plotting the thickened line with colored regions
    if orientation == 'vertical' or orientation == 'v':
        orientation = 'v'
    elif orientation == 'horizontal' or orientation == 'h':
        orientation = 'h'
    else:
        raise ValueError('Orientation can only be v/vertical/h/horizontal')
    if ax is None:
        if orientation == 'h':
            fig, ax = plt.subplots(figsize=(10, 2))
        elif orientation == 'v':
            fig, ax = plt.subplots(figsize=(2, 10))

    ticklabels = []
    tickpos = []

    if orientation == 'h':
        for i, (region, start, end) in enumerate(meta_region_rep):
            color = meta_region_color_map.get(region, 'white')
            ax.axvspan(start - 0.5, end + 0.5, ymin = 1 - fill_ratio, alpha=0.2, color=color)
        for i, (region, start, end) in enumerate(region_rep):
            if end - start < 10: continue
            tickpos.append((start+end) / 2)
            ticklabels.append(region)
            ax.axvline(end, 1 - fill_ratio, 1, color='black', linewidth=0.5)
        ax.axvline(0, 1 - fill_ratio, 1, color='black')
        ax.set_xticks(tickpos, ticklabels)
        # ax.set_xlim(0, 383)
        ax.set_yticks([], [])
    else: # ori == 'v'
        for i, (region, start, end) in enumerate(meta_region_rep):
            color = meta_region_color_map.get(region, 'white')
            ax.axhspan(start - 0.5, end + 0.5, xmin = 1 - fill_ratio, alpha=0.2, color=color)
        for i, (region, start, end) in enumerate(region_rep):
            if end - start < 10: continue
            tickpos.append((start+end) / 2)
            ticklabels.append(region)
            ax.axhline(end, 1 - fill_ratio, 1, color='black', linewidth=0.5)
        ax.set_yticks(tickpos, ticklabels)
        ax.axhline(0, 1 - fill_ratio, 1, color='black')
        # ax.set_ylim(0, 383)
        ax.set_xticks([], [])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    if show_figure:
        plt.tight_layout()
        plt.show()
    else:
        return ax
    
# Target brain regions for simplified classification
# TODO: this is buggy now since some region names are not in the tree
# TARGET_REGION_LIST: List[str] =  [
# "ACA", "AI", "BLA", "BMA", "BST", "DORpm", "DORsm", "HPC", "ILA", "LA", "LS", "LZ",
# "MBmot", "MBsta", "MEZ", "MOp", "MOs", "OLF", "ORB", "P", "PA", "PALd", "PALm", "PALv",
# "PIR", "PL", "PVR", "PVZ", "RSP", "SC", "SSp", "SSs", "STRd", "STRv", "sAMY", 
# "TH", 'HY', 'HPF', 'BS', 'CTX', 'CNU']

TARGET_REGION_LIST: List[str] =  ["TH", 'HY', 'HPF', 'BS', 'CTX', 'CNU']

# The TARGET REGION LIST should ensure that no former region will be the parent region for latter regions
# So we do a topology sort on them 
TARGET_REGION_LIST = sorted(TARGET_REGION_LIST, key = lambda x: len(tree.get_structures_by_acronym([x])[0]['structure_id_path'])) 

def get_meta_region_by_target_list(
    cluster_region_all: np.ndarray,
    target_region_list: List[str] = None
) -> str:
    """
    Map a brain region to one of the target meta-regions.
    
    This function checks if the given region belongs to any of the target
    regions by examining the structure hierarchy path.
    
    Parameters
    ----------
    cluster_region_all : np.ndarray
        Brain region acronym to classify.
    target_region_list : List[str], optional
        List of target region acronyms. If None, uses TARGET_REGION_LIST.
    
    Returns
    -------
    str
        Array of target region acronyms if found, otherwise 'other'.
    """

    if target_region_list is None:
        target_region_list = TARGET_REGION_LIST
    
    meta_region_all = []
    for region in cluster_region_all:
        path = tree.get_structures_by_acronym([region])[0]['structure_id_path']
        meta_region = None
        for target_region in target_region_list:
            region_id = tree.get_structures_by_acronym([target_region])[0]['id']
            if region_id in path:
                meta_region = target_region
                break
        # if not found in the target list, append a 'other'
        if meta_region is None:
            meta_region = 'other'
        meta_region_all.append(meta_region)
    return np.array(meta_region_all)

def get_meta_region_coarse(
    cluster_region_all: np.ndarray,
) -> np.ndarray:
    """
    Map brain region acronyms to their coarse meta-regions.
    First, get the region id path of the region.
        If path has 'Isocortex' or 'OLF', goes to depth 6 of the tree.
        If path has 'CA', goes to layer 8.
        If path has 'CTXsp', goes to layer 5.
        If path has 'DG', 'FC', or 'IG', goes to layer 7.
        If path has 'RHP', goes to layer 6.
        If path has 'CNU', goes to layer 3.
        If path has 'HY' or 'TH', goes to layer 5.
        If path has 'MB', goes to layer 4.
        If path has 'HB', goes to layer 4.
        If path has 'CB', goes to layer 3.
        Otherwise, goes to layer 1.
    If the region id path is shallower than the above, keep its region.

    Parameters
    ----------
        cluster_region_all : np.ndarray
        Array of brain region acronyms (e.g., ['VISp', 'CA1', 'TH']).
    Returns
    -------
    np.ndarray
        Array of meta-region names corresponding to each input region.
        Returns 'outside_brain' for regions not in the brain.
    """
    meta_region_all = []
    for region in cluster_region_all:
        if region == 'outside_brain' or region is None or region == "":
            meta_region_all.append('outside_brain')
            continue
        try:
            # Get structure id path
            structure = tree.get_structures_by_acronym([region])
            if not structure:
                meta_region_all.append('outside_brain')
                continue
            region_id_path = structure[0]['structure_id_path']
        except Exception:
            meta_region_all.append('outside_brain')
            continue

        # Get all IDs in the path except root (first entry is always 997 - 'root')
        path_ids = region_id_path[1:]

        # Find meta region by layer depth based on the rules
        meta_region = None

        structure_names = [tree.get_structures_by_id([rid])[0]['acronym'] for rid in path_ids]

        if 'Isocortex' in structure_names or 'OLF' in structure_names:
            depth = 6
        elif 'CA' in structure_names:
            depth = 8
        elif 'DG' in structure_names or 'FC' in structure_names or 'IG' in structure_names:
            depth = 7
        elif 'CTXsp' in structure_names:
            depth = 5
        elif 'RHP' in structure_names:
            depth = 6
        elif 'CNU' in structure_names:
            depth = 3
        elif 'HY' in structure_names or 'TH' in structure_names:
            depth = 5
        elif 'MB' in structure_names:
            depth = 4
        elif 'HB' in structure_names:
            depth = 4
        elif 'CB' in structure_names:
            depth = 3
        else:
            depth = 1

        # Choose the layer, as deep as available (may be truncated)
        if len(structure_names) >= depth:
            meta_region = structure_names[depth-1]
        elif structure_names:
            meta_region = structure_names[-1]
        else:
            meta_region = 'outside_brain'
        meta_region_all.append(meta_region)
    return np.array(meta_region_all)



def get_meta_region_IBL(
    cluster_region_all: np.ndarray,
    region_info_path : str = r"E:\Projects\SSA\AugusLabDP\utils\region_info.csv"
) -> np.ndarray:
    """
    Map brain region acronyms to their meta-regions.
    
    Converts specific brain region acronyms (e.g., 'VISp') to their
    higher-level meta-regions (e.g., 'Cerebrum', 'Cerebellum', 'Brain stem').
    
    Parameters
    ----------
    cluster_region_all : np.ndarray
        Array of brain region acronyms (e.g., ['VISp', 'CA1', 'TH']).
    region_info_path : str
        Path to the region info CSV file.
        DataFrame containing region info, should have at least the 'Beryl' column.
    
    Returns
    -------
    np.ndarray
        Array of meta-region names corresponding to each input region.
        Returns 'outside_brain' for regions not in the brain.
    """
    region_info = pd.read_csv(region_info_path)
    bottomline_meta_region = ['Isocortex', 'OLF', 'HPF', 'CNU', 'CTXsp', 'TH', 'HY', 'MB', 'HB', 'fiber tracts', 'VS']
    meta_region_all = []
    for region in cluster_region_all:
        if region == 'outside_brain':
            meta_region_all.append('outside_brain')
            continue
        region_id_path = tree.get_structures_by_acronym([region])[0]['structure_id_path']
        is_found = False
        for i in range(len(region_id_path)-1, -1, -1):
            region_name = tree.get_structures_by_id([region_id_path[i]])[0]['acronym']
            if region_name in region_info['Beryl'].values:
                meta_region_all.append(region_name)
                is_found = True
                break
            elif region_name in bottomline_meta_region:
                meta_region_all.append(region_name)
                is_found = True
                break
        if not is_found:
            meta_region_all.append('other')
    meta_region_all = np.array(meta_region_all)
    return meta_region_all
