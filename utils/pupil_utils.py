import numpy as np
import pandas as pd

def get_pupil_size(df_pupil, window_size = 1, frame_rate = 30):
    """ 
        From the DLC output, in the format of a dataframe, get the pupil size
        window_size: in seconds
        frame_rate: in Hz
    """
    from circle_fit import taubinSVD

    # In MultiIndex, replace 'pupil right' with 'pupilright' in the 2nd level of columns
    df_pupil.columns = [
        tuple(s.replace('pupil right', 'pupilright') if isinstance(s, str) else s for s in col)
        for col in df_pupil.columns
    ]
    model_name = df_pupil.columns[1][0]
    print(f"{len(df_pupil)} timepoints in total")
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
        confident_pupil_points = timepoint_likelihood[timepoint_likelihood > 0.85]
        if len(confident_pupil_points) == 4:
            xc, yc, r, sigma = taubinSVD(point_coordinates)
            # plot_circle(point_coordinates, xc, yc, r)
            r_all.append(r)
        elif len(confident_pupil_points) == 3:
            point_coordinates = point_coordinates[timepoint_likelihood > 0.85]
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
    return mean_pupil_size
