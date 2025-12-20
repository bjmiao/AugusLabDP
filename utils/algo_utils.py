"""Algorithm utilities for cross-correlation, RDM computation, and reduced rank regression."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def crosscorr(datax: pd.Series, datay: pd.Series, lag: int = 0) -> float:
    """
    Compute lag-N cross correlation between two pandas Series.
    
    Parameters
    ----------
    datax : pd.Series
        First time series data.
    datay : pd.Series
        Second time series data (must be same length as datax).
    lag : int, default 0
        Lag value for cross-correlation computation.
    
    Returns
    -------
    float
        Cross-correlation coefficient at the specified lag.
    """
    return datax.corr(datay.shift(lag))

def rdm(data: np.ndarray) -> np.ndarray:
    """
    Compute the representational dissimilarity matrix (RDM) for a given dataset.
    
    The RDM is computed as the pairwise Euclidean distances between all samples
    in the dataset.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array of shape (n_samples, n_features) representing the dataset.
    
    Returns
    -------
    np.ndarray
        The computed RDM matrix of shape (n_samples, n_samples) containing
        pairwise Euclidean distances.
    """
    # Compute pairwise distances between samples using vectorized operations
    distances = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1))
    
    return distances

class ReducedRankRegression:
    """
    Reduced Rank Regression (RRR) for predicting Y from X.
    
    This class implements regularized reduced rank regression. Note that X and Y
    should have their means subtracted before fitting.
    
    If rank is None, the rank is set to min(n_features, n_predictors) - 1.
    
    Prediction formula: Y_pred = X @ B
    
    Parameters
    ----------
    rank : Optional[int], default None
        Rank to compute reduced rank regression for. If None, uses
        min(n_features, n_predictors) - 1.
    lam : float, default 0
        Regularization parameter (currently not used in implementation).
    
    Attributes
    ----------
    rank : Optional[int]
        Rank of the regression.
    lam : float
        Regularization parameter.
    B : np.ndarray
        Regression coefficient matrix of shape (n_features, n_predictors).
    Vh_rrr : np.ndarray
        Right singular vectors from SVD decomposition.
    """
    
    def __init__(self, rank: Optional[int] = None, lam: float = 0) -> None:
        """
        Initialize ReducedRankRegression.
        
        Parameters
        ----------
        rank : Optional[int], default None
            Rank for reduced rank regression.
        lam : float, default 0
            Regularization parameter.
        """
        self.rank: Optional[int] = rank
        self.lam: float = lam
        self.B: Optional[np.ndarray] = None
        self.Vh_rrr: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fit the reduced rank regression model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features). Should be mean-centered.
        Y : np.ndarray
            Target data of shape (n_samples, n_predictors). Should be mean-centered.
        """
        if self.rank is None:
            self.rank = min(X.shape[1], Y.shape[1]) - 1

        # Compute ordinary least squares solution
        B_ols = np.linalg.inv(X.T @ X) @ X.T @ Y
        A = X @ B_ols
        
        # Perform SVD on A to get reduced rank representation
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        Vh_rrr = Vh[:self.rank, :]
        B_rrr = B_ols @ Vh_rrr.T @ Vh_rrr

        self.B = B_rrr
        self.Vh_rrr = Vh_rrr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Y from X using the fitted model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples, n_predictors).
        """
        if self.B is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return X @ self.B

def rrr_wrapper(
    spike_matrix: np.ndarray,
    video_motSVD: np.ndarray,
    rank: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function for reduced rank regression on spike and video motion data.
    
    This function performs RRR to predict spike matrix from video motion SVD components.
    Both inputs are mean-centered before fitting.
    
    Parameters
    ----------
    spike_matrix : np.ndarray
        Spike rate matrix of shape (n_timepoints, n_neurons).
    video_motSVD : np.ndarray
        Video motion SVD components of shape (n_timepoints, n_components).
    rank : Optional[int], default None
        Rank for reduced rank regression. If None, uses default rank.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - Vh_rrr: Right singular vectors from SVD (shape: (rank, n_neurons))
        - B: Regression coefficient matrix (shape: (n_components, n_neurons))
        - spike_pred: Predicted spike matrix (shape: (n_timepoints, n_neurons))
    """
    # Mean-center the video motion SVD components
    video_motSVD = video_motSVD - video_motSVD.mean(axis=0)[None, :]
    
    # Mean-center the spike matrix
    spike_matrix = spike_matrix - spike_matrix.mean(axis=0)[None, :]
    
    # Fit and predict using RRR
    rrr = ReducedRankRegression(rank=rank)
    rrr.fit(video_motSVD, spike_matrix)
    spike_pred = rrr.predict(video_motSVD)
    spike_residul = spike_matrix - spike_pred
    return rrr.Vh_rrr, rrr.B, spike_pred, spike_residul