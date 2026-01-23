"""Utilities package for neural data processing and analysis.

This package provides utilities for:
- Algorithm utilities (cross-correlation, RDM, reduced rank regression)
- Plotting utilities for neural data visualization
- Data readout utilities for loading and processing neural datasets
- Brain region mapping and visualization utilities
"""

from .algo_utils import (
    crosscorr,
    rdm,
    ReducedRankRegression,
    rrr_wrapper,
)

from .plot_utils import (
    create_colormap,
    plot_overall,
    SEED_COLORS,
)

from .readout_utils import (
    combine_time_bins,
    auguslab_manual_correct_ttl_button,
    auguslab_manual_create_experimental_tag,
    auguslab_manual_correct_ttl_camera,
    interpolate_array,
    get_cluster_region,
    load_dataset,
)

from .brain_region_utils import (
    mark_region_cluster,
    get_meta_region,
    plot_region_mark,
    get_meta_region_by_target_list,
    meta_region_color_map,
    condition_label_map,
    condition_color_map,
    TARGET_REGION_LIST,
)

__all__ = [
    # Algorithm utilities
    'crosscorr',
    'rdm',
    'ReducedRankRegression',
    'rrr_wrapper',
    # Plotting utilities
    'create_colormap',
    'plot_overall',
    'SEED_COLORS',
    # Readout utilities
    'combine_time_bins',
    'auguslab_manual_correct_ttl_button',
    'auguslab_manual_create_experimental_tag',
    'auguslab_manual_correct_ttl_camera',
    'interpolate_array',
    'get_cluster_region',
    'load_dataset',
    # Brain region utilities
    'mark_region_cluster',
    'get_meta_region',
    'plot_region_mark',
    'get_meta_region_by_target_list',
    'meta_region_color_map',
    'condition_label_map',
    'condition_color_map',
    'TARGET_REGION_LIST',
]

