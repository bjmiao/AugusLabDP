import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path
import allensdk
import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 25
# get home path
output_dir = os.path.join(str(Path.home()), "allen_reference_atlas")

rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 

def mark_region_cluster(arr):
    ''' Mark the boundary of regions
        Input: arr
        Output: [(repeat_element_1, start_1, end_1), (repeat_element_2, start_2, end_2)...] 
     '''
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


def get_meta_region(cluster_region_all):
    ''' From meta region (e.g. VISp -> Cereblum)'''
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

meta_region_color_map = {
    'Cerebrum': '#B0F0FF',
    'Cerebral cortex':'#B0FFB8',
    'Cerebral nuclei':'#98D6F9',

    'Brain stem':'#FF7080',
    'Interbrain':'#FF7080',
    'Midbrain':'#FF64FF',
    'Hindbrain':'#FF9B88',
    

    'Cerebellum':'#F0F080',
    'Cerebellar cortex':'#F0F080',
    'Cerebellar nuclei':'#F0F080'
}

condition_labep_map = {
    'iso': 'Isoflurane',
    'syncope': 'Syncope'
}
condition_color_map = {
    'iso': '#F89B50',
    'syncope': '#5B84C4',
}
def plot_region_mark(cluster_region, ax = None, orientation = 'v', reversed = None,
                     show_figure = True, meta_region_color_map = meta_region_color_map):
    if reversed is None:
        reversed = True if orientation == 'v' else 'False'
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
    if ax == None:
        if orientation == 'h':
            fig, ax = plt.subplots(figsize=(10, 2))
        elif orientation == 'v':
            fig, ax = plt.subplots(figsize=(2, 10))

    ticklabels = []
    tickpos = []

    if orientation == 'h':
        for i, (region, start, end) in enumerate(meta_region_rep):
            color = meta_region_color_map.get(region, 'white')
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, color=color)
        for i, (region, start, end) in enumerate(region_rep):
            tickpos.append((start+end) / 2)
            ticklabels.append(region)
            ax.axvline(end, color='black')
        ax.axvline(0, color='black')
        ax.set_xticks(tickpos, ticklabels)
        ax.set_xlim(0, 383)
        ax.set_yticks([], [])
    else: # ori == 'v'
        for i, (region, start, end) in enumerate(meta_region_rep):
            color = meta_region_color_map.get(region, 'white')
            ax.axhspan(start - 0.5, end + 0.5, alpha=0.3, color=color)
        for i, (region, start, end) in enumerate(region_rep):
            tickpos.append((start+end) / 2)
            ticklabels.append(region)
            ax.axhline(end, color='black')
        ax.set_yticks(tickpos, ticklabels)
        ax.axhline(0, color='black')
        ax.set_ylim(0, 383)
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
    
TARGET_REGION_LIST = [
    'TH', #thalamus
    'HY', # hypothalamus
    'HPF', # Hippocampal formation
    'BS', # brain stem
    'CTX', # cortex
    'CNU', # cerebral nuclei
]

def get_meta_region_by_target_list(region, target_region_list = TARGET_REGION_LIST):
    path = tree.get_structures_by_acronym([region])[0]['structure_id_path']
    for target_region in TARGET_REGION_LIST:
        region_id = tree.get_structures_by_acronym([target_region])[0]['id']
        if region_id in path:
            return target_region
    # if not among the list
    return 'other'

