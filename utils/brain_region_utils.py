"""Utilities for brain region mapping and visualization."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# Allen Brain Atlas reference space configuration
reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 25
# Get home path for cache directory
output_dir = os.path.join(str(Path.home()), "allen_reference_atlas")

# Initialize reference space cache
rspc = ReferenceSpaceCache(
    resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json'
)
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 

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
        if region == 'outside_brain':
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
    'Cerebellar nuclei': '#F0F080'
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
    region: str,
    target_region_list: List[str] = None
) -> str:
    """
    Map a brain region to one of the target meta-regions.
    
    This function checks if the given region belongs to any of the target
    regions by examining the structure hierarchy path.
    
    Parameters
    ----------
    region : str
        Brain region acronym to classify.
    target_region_list : List[str], optional
        List of target region acronyms. If None, uses TARGET_REGION_LIST.
    
    Returns
    -------
    str
        Target region acronym if found, otherwise 'other'.
    """
    if target_region_list is None:
        target_region_list = TARGET_REGION_LIST
    
    path = tree.get_structures_by_acronym([region])[0]['structure_id_path']
    for target_region in target_region_list:
        region_id = tree.get_structures_by_acronym([target_region])[0]['id']
        if region_id in path:
            return target_region
    # If not among the target list
    return 'other'

 