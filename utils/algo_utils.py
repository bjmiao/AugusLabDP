import numpy as np
import pandas as pd


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def rdm(data):
    """
    Compute the representational dissimilarity matrix (RDM) for a given dataset.

    Args:
        data (numpy.ndarray): A 2D numpy array of shape (n_samples, n_features)
            representing the dataset.

    Returns:
        numpy.ndarray: The computed RDM matrix of shape (n_samples, n_samples).
    """
    # Compute pairwise distances between samples using vectorized operations
    distances = np.sqrt(np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1))
    
    return distances

# Then we do reduced rank regression to get the motion independent component
class ReducedRankRegression:
    """predict Y from X using regularized reduced rank regression
    *** subtract mean from X and Y before predicting
    if rank is None, returns A and B of full-rank (minus one) prediction

    Prediction:
    >>> Y_pred = X @ B @ A.T
    Parameters
    ----------
    X : 2D array, input data, float32 torch tensor (n_samples, n_features)
    Y : 2D array, data to predict, float32 torch tensor (n_samples, n_predictors)
    rank : int (optional, default None)
        rank to compute reduced rank regression for
    lam : float (optional, default 0)
        regularizer
    Returns
    """
    def __init__(self, rank = None, lam = 0):
        self.rank = rank
        self.lam = lam
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.rank is None:
            self.rank = min(self.X.shape[1], self.Y.shape[1]) - 1

        B_ols = np.linalg.inv(X.T @ X) @ X.T @ Y
        A = X @ B_ols
        # perform svd on A
        U, S, Vh = np.linalg.svd(A, full_matrices = False)
        Vh_rrr = Vh[:self.rank, :]
        B_rrr = B_ols @ Vh_rrr.T @ Vh_rrr

        self.B = B_rrr
        self.Vh_rrr = Vh_rrr
        # Y_pred = X @ B_rrr
        # return Vh_rrr, B_rrr, Y_pred

    def predict(self, X):
        return X @ self.B

def rrr_wrapper(spike_matrix, video_motSVD, rank = None):
    # video_motSVD = torch.Tensor(video_motSVD)
    video_motSVD = video_motSVD - video_motSVD.mean(axis = 0)[None, :]
    # spike_matrix = torch.Tensor(spike_matrix)
    spike_matrix = spike_matrix - spike_matrix.mean(axis = 0)[None, :]
    rrr = ReducedRankRegression(rank = rank)
    rrr.fit(video_motSVD, spike_matrix)
    spike_pred = rrr.predict(video_motSVD)
    spike_residul = spike_matrix - spikes_pred
    return rrr.Vh_rrr, rrr.B, spike_pred