"""Plotting utilities for neural data visualization."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


from readout_utils import combine_time_bins
# Seed colors for trajectory visualization
# Each tuple contains (start_color, end_color) for a time period
SEED_COLORS: List[Tuple[str, str]] = [
    ("#0501ff", "#00ffff"),
    ("#FF0000", "#ff7b00"),
    # ("#000000", "#444444"),  # Commented out colors
    # ("#00ff48", "#d0ff00"),
    ("#3476cc", "#6696ff"),
    ("#f700ff", "#ff3f04"),
    ("#2D6141", "#3F7550"),
    ("#ff9500", "#ffe83d"),
]


def create_colormap(time_periods: List[Tuple[float, float]]) -> LinearSegmentedColormap:
    """
    Create a colormap that maps key points of a trajectory.
    
    The time_periods should be in the format of tuples:
    [(0, 1000), (1000, 1500), (1500, 2000), (2000, 4701)]
    
    Each period is assigned a seed color from SEED_COLORS, and colors
    transition linearly between periods.
    
    Parameters
    ----------
    time_periods : List[Tuple[float, float]]
        List of (start_time, stop_time) tuples defining time periods.
    
    Returns
    -------
    LinearSegmentedColormap
        A custom colormap for trajectory visualization.
    """
    colors: List[Tuple[float, str]] = []
    max_time = np.max([x[1] for x in time_periods])
    
    for index, (start, stop) in enumerate(time_periods):
        seed_color_start, seed_color_stop = SEED_COLORS[index % len(SEED_COLORS)]
        colors += [
            (start / max_time, seed_color_start),
            (stop / max_time, seed_color_stop)
        ]
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return cmap


def plot_overall(
    spikes_matrix_all: np.ndarray,
    ttl_button: np.ndarray,
    ttl_camera: np.ndarray,
    video_motion: np.ndarray,
    experimental_label_tag: Dict[str, Tuple[float, float]],
    combine_timpoints: int
) -> Tuple[Figure, Axes]:
    """
    Create a comprehensive overview plot of neural data analysis.
    
    This function generates a multi-panel figure including:
    - 3D PCA trajectory of spike data
    - Pairwise group distance heatmap
    - Cosine similarity matrix
    - PSTH (Peri-Stimulus Time Histogram)
    - PSTH difference
    - Traveling speed in PCA space
    - Motion energy overlay
    - TTL button and camera signals
    
    Parameters
    ----------
    spikes_matrix_all : np.ndarray
        Spike rate matrix of shape (n_timepoints, n_neurons).
    ttl_button : np.ndarray
        TTL button signal array.
    ttl_camera : np.ndarray
        TTL camera signal array.
    video_motion : np.ndarray
        Video motion energy array.
    experimental_label_tag : Dict[str, Tuple[float, float]]
        Dictionary mapping experimental period names to (start_time, stop_time) tuples.
    combine_timpoints : int
        Number of timepoints to combine for binning.
    
    Returns
    -------
    Tuple[Figure, Axes]
        A tuple containing the figure and the 3D axes object.
    """
    experimental_cmap = create_colormap(list(experimental_label_tag.values()))
    timepoints = spikes_matrix_all.shape[0]
    pca = PCA(n_components=3)
    # Combine time bins and compute time resolution
    time_resolution = 0.1 * combine_timpoints  # in seconds
    spikes_matrix_all_combined = combine_time_bins(spikes_matrix_all, combine_timpoints)
    x_time_axis = combine_time_bins(np.arange(spikes_matrix_all.shape[0]), combine_timpoints)
    # Perform PCA on combined spike matrix
    spikes_pca = pca.fit_transform(spikes_matrix_all_combined)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(22, 15))
    gs = GridSpec(5, 5, width_ratios=[2, 2.5, 1, 1, 1])
    
    # 3D PCA trajectory plot
    ax_3d = fig.add_subplot(gs[0:3, 0:2], projection='3d')
    p = ax_3d.scatter(
        spikes_pca[:, 0], spikes_pca[:, 1], spikes_pca[:, 2],
        s=1, cmap=experimental_cmap, c=np.arange(x_time_axis.shape[0])
    )
    ax_3d.set_title(f"Time bin = {time_resolution}s")

    # Groupwise distance heatmap
    ax_groupwise_distance = fig.add_subplot(gs[3:, 0])
    # Calculate the distance in PCA space
    # Step 1: Get the centroid for each experimental period
    centroid: Dict[str, np.ndarray] = {}
    centroid['Origin'] = np.array([0, 0, 0])
    for period_tag, (start_time_sec, stop_time_sec) in experimental_label_tag.items():
        start_index = int(start_time_sec / combine_timpoints * 10)
        stop_index = int(stop_time_sec / combine_timpoints * 10)
        centroid[period_tag] = np.mean(spikes_pca[start_index:stop_index, :], axis = 0)
    mean_distance_matrix = np.zeros((len(centroid), len(centroid)))
    std_distance_matrix = np.zeros((len(centroid), len(centroid)))
    annot_matrix = np.empty((len(centroid), len(centroid)), dtype = object)
    period_tags = list(centroid.keys())
    for i, p1 in enumerate(period_tags):
        for j, p2 in enumerate(period_tags):
            if p1 == 'Origin' and p2 == 'Origin':
                mean, std = 0, 0
            elif p1 == 'Origin' or p2 == 'Origin':
                if p1 == 'Origin':
                    (start_time_sec, stop_time_sec) = experimental_label_tag[p2]
                if p2 == 'Origin':
                    (start_time_sec, stop_time_sec) = experimental_label_tag[p1]
                start_index = int(start_time_sec / combine_timpoints * 10)
                stop_index = int(stop_time_sec / combine_timpoints * 10)
                points = spikes_pca[start_index:stop_index, :]
                dist = np.sqrt(np.sum(np.square(points), axis = 1))
                mean = np.mean(dist)
                std = np.std(dist)
            else:  # Between two groups, calculate distance from p1 points to p2's centroid
                (start_time_sec, stop_time_sec) = experimental_label_tag[p1]
                start_index = int(start_time_sec / combine_timpoints * 10)
                stop_index = int(stop_time_sec / combine_timpoints * 10)
                points_p1 = spikes_pca[start_index:stop_index, :]

                centroid_p2 = centroid[p2]
                dist = np.sqrt(np.sum(np.square(points_p1 - centroid_p2[None, :]), axis = 1))
                mean = np.mean(dist)
                std = np.std(dist)
            mean_distance_matrix[i, j] = mean
            std_distance_matrix[i, j] = std
            annot_matrix[i, j] = f"{mean:.2f}\n(±{std:.2f})"
    sns.heatmap(mean_distance_matrix, annot = annot_matrix, fmt = '',
                square = True, cmap = 'Reds', ax = ax_groupwise_distance, xticklabels=period_tags, yticklabels=period_tags)
    ax_groupwise_distance.set_title("Pairwise group distance")

    # Cosine Similarity Pattern
    ax_similarity_matrix = fig.add_subplot(gs[3:, 1])
    # Combine spikes for similarity computation (downsample to ~1000 timepoints)
    spike_matrix_combined = combine_time_bins(
        spikes_matrix_all, spikes_matrix_all.shape[0] // 1000
    )
    # Compute similarity between all pairs of timepoints
    timepoint_similarity = cosine_similarity(spike_matrix_combined)
    sns.heatmap(timepoint_similarity, ax=ax_similarity_matrix, vmin=0, vmax=1)

    # PSTH (Peri-Stimulus Time Histogram) - mean firing rate over time
    ax_psth = fig.add_subplot(gs[0, 2:])
    psth = spikes_matrix_all_combined.mean(axis=1)
    ax_psth.scatter(
        x_time_axis, psth, cmap=experimental_cmap,
        c=np.arange(x_time_axis.shape[0]), s=1
    )
    ax_psth.set_ylabel("Mean Firing Rate")
    # Add vertical lines at experimental period boundaries
    for _, stop_time in experimental_label_tag.values():
        ax_psth.axvline(stop_time * 10, linewidth=0.5, color='black')

    # PSTH difference - rate of change in firing rate
    ax_psth_diff = fig.add_subplot(gs[1, 2:])
    ax_psth_diff.plot(x_time_axis, np.diff(psth, prepend=0), linewidth=0.3)
    ax_psth_diff.set_ylabel("Mean Firing Rate Difference")
    for _, stop_time in experimental_label_tag.values():
        ax_psth_diff.axvline(stop_time * 10, linewidth=0.5, color='black')

    # Traveling speed in PCA space
    ax_travel = fig.add_subplot(gs[2, 2:])
    trajectory_distance = np.zeros(spikes_pca.shape[0])
    # Compute Euclidean distance between consecutive timepoints in PCA space
    trajectory_distance[1:] = np.sum(
        np.square(spikes_pca[:-1, :] - spikes_pca[1:, :]), axis=1
    )
    ax_travel.plot(x_time_axis, trajectory_distance, color='black', linewidth=0.3)
    ax_travel.set_ylabel("Travelling speed in PCA space")
    for _, stop_time in experimental_label_tag.values():
        ax_travel.axvline(stop_time * 10, linewidth=0.5, color='black')

    # Motion energy overlay on same axes
    ax_motion_energy = ax_travel.twinx()
    video_motion_combine = combine_time_bins(
        video_motion[:, None], combine_timpoints
    )[:, 0]
    ax_motion_energy.plot(x_time_axis, video_motion_combine, linewidth=0.3, color='red')
    ax_motion_energy.set_ylabel("Motion energy")
    # Compute and display correlation between trajectory distance and motion energy
    correlation = np.corrcoef(video_motion_combine, trajectory_distance)[0, 1]
    ax_travel.set_title(f"Correlation = {correlation:.2f}")

    # TTL Button signal
    ax_ttl_button = fig.add_subplot(gs[3, 2:])
    ttl_button_for_plot = combine_time_bins(
        ttl_button[:, None], len(ttl_button) // len(x_time_axis)
    )[:, 0]
    ttl_button_for_plot = ttl_button_for_plot[:len(x_time_axis)]
    ax_ttl_button.plot(x_time_axis, ttl_button_for_plot)
    for _, stop_time in experimental_label_tag.values():
        ax_ttl_button.axvline(stop_time * 10, linewidth=0.5, color='black')
    ax_ttl_button.set_ylabel("TTL Button")

    # TTL Camera signal
    ax_ttl_camera = fig.add_subplot(gs[4, 2:])
    ttl_camera_for_plot = combine_time_bins(
        ttl_camera[:, None], len(ttl_camera) // len(x_time_axis)
    )[:, 0]
    ttl_camera_for_plot = ttl_camera_for_plot[:len(x_time_axis)]
    ax_ttl_camera.plot(x_time_axis, ttl_camera_for_plot)
    for _, stop_time in experimental_label_tag.values():
        ax_ttl_camera.axvline(stop_time * 10, linewidth=0.5, color='black')
    ax_ttl_camera.set_ylabel("TTL Camera")

    fig.tight_layout()
    return fig, ax_3d