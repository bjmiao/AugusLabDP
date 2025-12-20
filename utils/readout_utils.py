import os
import numpy as np
import json
import pandas as pd

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
        need_flatten = True
        matrix = matrix[:, None]
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


def auguslab_manual_correct_ttl_button(nidq_ttl_button, nidq_sampling_rate, session_name, session_type):
    '''
        For specific sessions given session name, we manually change the value of nidq_ttl_camera_array.
        We make in-place change to the nidq_ttl_button
    ''' 
    if session_type == 'iso_day1' or session_type == 'iso_day2':
        # get the up-edge for the button (TTL laser signal for syncope)
        nidq_ttl_button_up = np.where( (nidq_ttl_button[:-1] == 1) & (nidq_ttl_button[1:] == 0))[0] 
        nidq_ttl_button[nidq_ttl_button_up[0]:nidq_ttl_button_up[-1]] = 1
    elif session_name == 'test2':
        pass
    return

def auguslab_manual_create_experimental_tag(results, session_name, session_type):
    session_start_time, session_stop_time = results['session_start_time'], results['session_stop_time']
    # only one ttl square wave, meaning 
    nidq_ttl_button = results['ttl_button']
    # get the up-edge for the button (TTL laser signal for syncope)
    nidq_ttl_button_up = np.where( (nidq_ttl_button[:-1] == 1) & (nidq_ttl_button[1:] == 0))[0]
    nidq_sampling_rate = float(results['ttl_meta']['niSampRate'])
    nidq_ttl_button_high_time = np.where(nidq_ttl_button == 1)[0] / nidq_sampling_rate
    nidq_ttl_button_up_time = nidq_ttl_button_up / nidq_sampling_rate
    if session_type == 'syncope':
        # find 5 big chunks
        threshold = 10 # in s, can be any value between 1 and 300
        stim_train_start = [nidq_ttl_button_up_time[0]] + nidq_ttl_button_up_time[np.where(nidq_ttl_button_up_time[1:] - nidq_ttl_button_up_time[:-1] > threshold)[0] + 1].tolist()
        stim_train_end = nidq_ttl_button_up_time[np.where(nidq_ttl_button_up_time[1:] - nidq_ttl_button_up_time[:-1] > threshold)[0]].tolist() + [nidq_ttl_button_up_time[-1]]
        stim_train_start, stim_train_end = np.array(stim_train_start), np.array(stim_train_end)
        assert len(stim_train_start) == 5
        assert len(stim_train_end) == 5

        # correct for start_time and stop_time
        stim_train_start = stim_train_start - session_start_time
        stim_train_end = stim_train_end - session_start_time
        tags = ["Baseline", "Laser(Phony)", "Recover 1", "Laser(5Hz)", "Recover 2", "Laser(10Hz)", "Recover 3", "Laser(20Hz)", "Recover 4", "Laser (20Hz,L)", "Recover 5"]
        all_timelabels = [[start, stop] for start, stop in zip(stim_train_start, stim_train_end)]
        all_timelabels = [x for xs in all_timelabels for x in xs] # flatten
        period_start_timetag = [0] + all_timelabels
        period_end_timetag = all_timelabels + [session_stop_time - session_start_time]
        # print(period_start_timetag, period_end_timetag)
        experimental_label_tag = {}
        for tag, period_start, period_end in zip(tags, period_start_timetag, period_end_timetag):
            experimental_label_tag[tag] = (period_start, period_end)
    elif session_type == 'ketamine':
        ketamine_onset = nidq_ttl_button_up_time[0] - session_start_time # correct for camera
        experimental_label_tag = {'Baseline': (0, ketamine_onset),
                    'Ketamine': (ketamine_onset, ketamine_onset+120),
                    'Recover': (ketamine_onset+120, session_stop_time - session_start_time)}
    elif session_type == 'iso_day1' or session_type == 'iso_day2':
        # only one ttl square wave
        iso_onset = nidq_ttl_button_high_time[0] - session_start_time
        iso_offset = nidq_ttl_button_high_time[-1] - session_start_time
        tag = 'Iso'
        if session_type == 'iso_day1': tag += '(5%)'
        if session_type == 'iso_day2': tag += '(1.5%)'
          
        experimental_label_tag = {'Baseline': (0, iso_onset),
                                  tag:(iso_onset, iso_offset),
                                  'Recover':(iso_offset, session_stop_time - session_start_time)}
    elif session_type == 'oxy_iso_day1' or session_type == 'oxy_iso_day2':
        # one ttl pulse (turning on oxy)
        oxy_on_time = nidq_ttl_button_up_time[0] - session_start_time
        threshold = 100 # in s, can be any number between 10 to 600
        # iso_onset_index = np.where(nidq_ttl_button_up_time[1:] - nidq_ttl_button_up_time[:-1] > threshold)[0][0] # first such element

        iso_onset_time = nidq_ttl_button_up_time[1] - session_start_time
        iso_offset_time = nidq_ttl_button_high_time[-1] - session_start_time
        tag = 'Iso'
        if session_type == 'oxy_iso_day1': tag += '(5%)'
        if session_type == 'oxy_iso_day2': tag += '(1.5%)'
        experimental_label_tag = {'Baseline' : (0, oxy_on_time),
                                  'Oxygen': (oxy_on_time, iso_onset_time),
                                  tag:(iso_onset_time, iso_offset_time),
                                  'Recover':(iso_offset_time, session_stop_time - session_start_time)}

    elif session_type == '52N_D4':
        # TODO: correct for the caera
        experimental_label_tag = {'Baseline': (0, 600),
                                'Oxygen': (600, 2400),
                                'Iso 5%': (2400, 2460),
                                'Recover1': (2460, 4200),
                                'Iso 1.5%': (4200, 6000),
                                'Recover2': (6000, 7227.3)}
    return experimental_label_tag

def auguslab_manual_correct_ttl_camera(nidq_ttl_camera, nidq_sampling_rate, session_name, session_type):
    '''
        For specific sessions given session name, we manually change the value of nidq_ttl_camera_array.
        H
        We make in-place change to the nidq_ttl_camera
    '''
    if session_name == 'test':
        pass
    elif session_name == 'test2':
        pass
    return


def interpolate_array(arr, target_size):
    """
    Interpolate array from its current size to target size using linear interpolation
    """
    current_size = len(arr)
    
    # Create indices for the original and target arrays
    old_indices = np.linspace(0, current_size - 1, current_size)
    new_indices = np.linspace(0, current_size - 1, target_size)
    
    # Perform linear interpolation
    interpolated = np.interp(new_indices, old_indices, arr)
    
    return interpolated

def get_cluster_region(template_depth, probe_depth_table):
    """
    Notice: the template_depth is tip (0) to base (max)
        while the probe depth table is surface (0) to deep (max).
        We consider the depth 0 is the deepset if probe_depth_table.
     
      template_depth: depth for each cluster (each unit)
      probe_depth_table: depth table from the matlab file, a dataframe
    """

    
    # transform template_depth to depth on the probe
    total_depth = probe_depth_table['end_depth'].max()
    template_depth = total_depth - template_depth

    # find the region for each unit
    cluster_region = []
    for depth in template_depth:
        l = np.where(probe_depth_table['start_depth'] <= depth)[0]
        if len(l) == 0: # not included in the depth table, label root (ID=1)
            region_id = 'root' # 0 is outside the brain
        else:
            region_id = probe_depth_table['acronym'][l[-1]] 
        cluster_region.append(region_id)
    return cluster_region

def load_dataset(data_folder, session_name, session_type, probe = 'all', probe_mapping = {}, need_modules = ['spike', 'video', 'ttl', 'eeg', 'ecg']):
    session_folder = os.path.join(data_folder, session_name)
    results = {}
    # load TTL first
    if 'ttl' in need_modules:
        try:
            nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
            results['has_ttl_meta'] = True
            results['ttl_meta'] = nidq_meta
        except FileNotFoundError as e:
            results['has_ttl_meta'] = False
        try:
            nidq_ttl_camera = np.load(os.path.join(data_folder, session_name, "nidq_TTL_camera.npy"))
            results['has_ttl_camera'] = True
            # Go through a lab-specific manual correction for camera TTL signal due to signal loss in some sessions
            auguslab_manual_correct_ttl_camera(nidq_ttl_camera, float(nidq_meta['niSampRate']), session_name, session_type)
            results['ttl_camera'] = nidq_ttl_camera
            nidq_ttl_camera_high = np.where(nidq_ttl_camera == 1)[0]
            session_start_time = nidq_ttl_camera_high[0] / float(nidq_meta['niSampRate'])
            session_stop_time = nidq_ttl_camera_high[-1] / float(nidq_meta['niSampRate'])
            results['session_start_time'] = session_start_time
            results['session_stop_time'] = session_stop_time
        except FileNotFoundError as e:
            results['has_ttl_camera'] = False
            results['session_start_time'] = 0
            results['session_stop_time'] = len(nidq_ttl_camera) / float(nidq_meta['niSampRate'])
        try:
            nidq_ttl_button = np.load(os.path.join(data_folder, session_name, "nidq_TTL_button.npy"))
            results['has_ttl_button'] = True
            auguslab_manual_correct_ttl_button(nidq_ttl_button, float(nidq_meta['niSampRate']), session_name, session_type)
            results['ttl_button'] = nidq_ttl_button
            experimental_label_tag = auguslab_manual_create_experimental_tag(results, session_name, session_type)
            results['experimental_label_tag'] = experimental_label_tag
        except FileNotFoundError as e:
            results['has_ttl_button'] = False
    else:
        results['has_ttl_meta'] = False
        results['has_ttl_camera'] = False
        results['has_ttl_button'] = False

    # Load spiking matrix first, since it decides the shape of the matrix
    if 'spike' in need_modules:
        all_probes = set()
        spike_timebin = 0.1 # in s
        if probe == 'all':
            for f in os.scandir(session_folder):
                if f.is_dir():
                    all_probes.add(f.path)
            all_spike_matrix = []
            all_cluster_region = []
            for probe_path in all_probes:
                probe_name = os.path.basename(probe_path)
                spike_matrix = np.load(os.path.join(session_folder, probe_path, "spike_rate_matrix_100ms.npy"))
                all_spike_matrix.append(spike_matrix)

                # try get the cluster region
                templateDepths = np.load(os.path.join(session_folder, probe_path, "templateDepths.npy"))
                probe_index = probe_mapping.get(probe_name, None)
                if probe_index is not None:
                    try:
                        df_depth_table = pd.read_csv(os.path.join(session_folder, f"probe_{probe_index}_location.csv"))
                        results['depth_table'] = df_depth_table
                        cluster_region = get_cluster_region(templateDepths, df_depth_table)
                        all_cluster_region.append(cluster_region)
                    except Exception as e:
                        print(e)
                        results['has_cluster_region'] = False
            align_timestep = np.min([matrix.shape[1] for matrix in all_spike_matrix])
            all_spike_matrix = [matrix[:, :align_timestep] for matrix in all_spike_matrix]
            all_spike_matrix = np.concatenate(all_spike_matrix, axis = 0)
            results['spike_matrix'] = all_spike_matrix
            if results.get('has_cluster_region', True):
                all_cluster_region = np.concatenate(all_cluster_region)
                results['cluster_region'] = all_cluster_region
                results['has_cluster_region'] = True
        else:
            raise NotImplementedError

        # If TTL camera is presented, correct the spike matrix by the video time
        if results['has_ttl_camera']:
            session_start_index = int(results['session_start_time'] / spike_timebin)
            session_stop_index = int(results['session_stop_time']  / spike_timebin)
            results['spike_matrix'] = results['spike_matrix'][:, session_start_index:session_stop_index].T
        results['has_spike'] = True
    else:
        results['has_spike'] = False
        results['has_cluster_region'] = False
    if 'video' in need_modules:
        try:
            video_motion = np.load(os.path.join(session_folder, "face_motion.npy"))
            video_motSVD = np.load(os.path.join(session_folder, "face_motSVD.npy"))
            results['has_video'] = True
            spike_matrix = results['spike_matrix']
            motSVD_interpolated_matrix = np.empty((spike_matrix.shape[0], video_motSVD.shape[1]))
            for i in range(video_motSVD.shape[1]):
                motSVD_interpolated_matrix[:, i] = interpolate_array(video_motSVD[:, i], spike_matrix.shape[0])
            video_motion_interpolated = interpolate_array(video_motion, spike_matrix.shape[0])
            results['video_motSVD'] = motSVD_interpolated_matrix
            results['video_motion'] = video_motion_interpolated
        except FileNotFoundError as e:
            results['has_video'] = False
    else:
        results['has_video'] = False
    if 'eeg' in need_modules:
        try:
            nidq_eeg = np.load(os.path.join(session_folder, 'nidq_EEG.npy'))
            if results['has_ttl_camera']:
                nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
                sampFrequency = float(nidq_meta['niSampRate'])
                session_start_index = int(results['session_start_time'] * sampFrequency)
                session_stop_index = int(results['session_stop_time']  * sampFrequency)
                nidq_eeg = nidq_eeg[session_start_index:session_stop_index]
            results['eeg'] = nidq_eeg
        except FileNotFoundError as e:
            results['has_nidq_eeg'] = False
    else:
        results['has_nidq_eeg'] = False
    if 'ecg' in need_modules:
        try:
            nidq_ecg = np.load(os.path.join(session_folder, 'nidq_ECG.npy'))
            if results['has_ttl_camera']:
                nidq_meta = json.load(open(os.path.join(data_folder, session_name, "nidq_meta.json")))
                sampFrequency = float(nidq_meta['niSampRate'])
                session_start_index = int(results['session_start_time'] * sampFrequency)
                session_stop_index = int(results['session_stop_time']  * sampFrequency)
                nidq_ecg = nidq_ecg[session_start_index:session_stop_index]
            results['ecg'] = nidq_ecg
        except FileNotFoundError as e:
            results['has_nidq_ecg'] = False
    else:
        results['has_nidq_ecg'] = False
    return results