import numpy as np
def combine_time_bins(matrix, bin_size=10):
    """
    Combine time bins in a matrix with shape (#timestep, #neuron).
    
    Parameters:
    - matrix: numpy array with shape (#timestep, #neuron)
    - bin_size: number of time bins to combine (default: 10)
    
    Returns:
    - combined_matrix: numpy array with reduced number of timesteps
    """
    need_flatten = False
    if len(matrix.shape) == 1:
        matrix = matrix[:, None]
        need_flatten = True
    num_timesteps, num_neurons = matrix.shape
    num_full_bins = num_timesteps // bin_size
    
    # Reshape and combine full bins
    reshaped = matrix[:num_full_bins*bin_size].reshape(num_full_bins, bin_size, num_neurons)
    combined = np.mean(reshaped, axis=1)
    
    # Handle remaining timesteps if any
    if num_timesteps % bin_size != 0:
        remaining = matrix[num_full_bins*bin_size:]
        remaining_mean = np.mean(remaining, axis=0, keepdims=True)
        combined = np.vstack((combined, remaining_mean))
    
    if need_flatten:
        combined = combined.flatten()
    return combined
