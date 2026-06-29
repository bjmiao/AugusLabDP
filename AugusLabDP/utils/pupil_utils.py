import numpy as np
import pandas as pd

__all__ = ['get_pupil_size']

def get_pupil_size(df_pupil, window_size = 1, frame_rate = 30, max_nan_time = 3):
    """ 
        From the DLC output, in the format of a dataframe, get the pupil size
        window_size: in seconds
        frame_rate: in Hz
        max_nan_time: in seconds, below this time nan value will be interpolated
    """
    from circle_fit import taubinSVD

    # In MultiIndex, replace 'pupil right' with 'pupilright' in the 2nd level of columns
    df_pupil.columns = [
        tuple(s.replace('pupil right', 'pupilright') if isinstance(s, str) else s for s in col)
        for col in df_pupil.columns
    ]
    model_name = df_pupil.columns[1][0]
    # print(f"{len(df_pupil)} timepoints in total")
    likelihoods = []
    arrs = {}
    for bodypart in ['pupiltop', 'pupilbot', 'pupilleft', 'pupilright']:
        x = df_pupil[(model_name, bodypart, "x")]
        y = df_pupil[(model_name, bodypart, "y")]
        likelihood = df_pupil[(model_name, bodypart, "likelihood")]
        arrs[bodypart] = np.array([x, y]).T
        likelihoods.append(likelihood)
    likelihoods = np.array(likelihoods).T

    r_all = []
    for timepoint in range(len(df_pupil)):
        point_coordinates = np.array([arrs['pupiltop'][timepoint, :], arrs['pupilbot'][timepoint, :], arrs['pupilleft'][timepoint, :], arrs['pupilright'][timepoint, :]])
        timepoint_likelihood = likelihoods[timepoint, :]
        # if all likelihoods are above 0.85, then use the circle fit
        confident_pupil_points = timepoint_likelihood[timepoint_likelihood > 0.9]
        if len(confident_pupil_points) == 4:
            xc, yc, r, sigma = taubinSVD(point_coordinates)
            # plot_circle(point_coordinates, xc, yc, r)
            r_all.append(r)
        elif len(confident_pupil_points) == 3:
            point_coordinates = point_coordinates[timepoint_likelihood > 0.9]
            xc, yc, r, sigma = taubinSVD(point_coordinates)
            # plot_circle(point_coordinates, xc, yc, r)
            r_all.append(r)
        else:
            r_all.append(np.nan)
    mean_pupil_size = []
    window_size = int(window_size * frame_rate)
    for i in range(0, len(r_all), window_size):
        mean_pupil_size.append(np.nanmean(r_all[i:i+window_size]))
    mean_pupil_size = np.array(mean_pupil_size)

    
    # Replace runs of NaNs shorter than max_nan_time seconds, otherwise leave as NaN
    def fill_short_nan_runs_linear(arr, max_run): # max run in num windows
        isnan = np.isnan(arr)
        n = len(arr)

        i = 0
        while i < n:
            if isnan[i]:
                # Start of a NaN run
                start = i
                while i < n and isnan[i]:
                    i += 1
                end = i  # arr[start:end] are NaNs; arr[start-1] and arr[end] are non-NaN if within bounds

                run_length = end - start
                print(f"run_length: {run_length} ({start} to {end})")

                if run_length < max_run:
                    left_val = arr[start - 1] if start > 0 else arr[end] if end < n else np.nan
                    right_val = arr[end] if end < n else arr[start - 1] if start > 0 else np.nan

                    # If both left and right are available, do linear interpolation
                    if (start > 0) and (end < n):
                        arr[start:end] = np.linspace(left_val, right_val, run_length + 2)[1:-1]
                        print(f"Filled {run_length} NaNs", left_val, right_val)
                    elif start == 0 and end < n:
                        arr[start:end] = right_val
                    elif start > 0 and end == n:
                        arr[start:end] = left_val
                    # else: both ends out of bounds; should rarely happen
                # else: leave NaNs as is
            else:
                i += 1
    # max_run = max(int(max_nan_time / window_size), 1)
    max_run = 300 # TODO: har code, fill all NANs
    fill_short_nan_runs_linear(mean_pupil_size, max_run = max_run)

    if np.any(np.isnan(mean_pupil_size)):
        print(f"There are still {np.sum(np.isnan(mean_pupil_size))} NaNs in the pupil size")
    return mean_pupil_size