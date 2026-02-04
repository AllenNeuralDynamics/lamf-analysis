import numpy as np

## About running
def total_run_distance(bsd):
    ''' Calcualte total run distance from BehavrioSessionDataset object
    Returns in meters
    '''
    running_df = bsd.running_speed
    total_distance = running_df.speed.sum() * np.median(np.diff(bsd.running_speed.timestamps)) / 100  # in m/s
    return total_distance


def process_running_speed(bsd,
                          mean_window_size: float = 2.0, # seconds
                          std_window_size: float = 1.0, # seconds
                          run_threshold: float = 5.0,  # speed threshold for running
                          jitter_threshold: float = 0.5,  # std threshold for jittering
                          ):
    running_df = bsd.running_speed
    recording_rate = 1 / np.median(np.diff(running_df['timestamps']))
    mean_samples = int(recording_rate * mean_window_size)
    std_samples = int(recording_rate * std_window_size)
    # Calculate rolling features
    running_df['roll_mean'] = running_df['speed'].rolling(mean_samples,
                                                        center=True).mean()
    running_df['roll_std'] = running_df['speed'].rolling(std_samples,
                                                        center=True).std()

    # Logic for Detection
    # 1. Running: Mean speed is high
    running_df['is_running'] = running_df['roll_mean'] > run_threshold

    # 2. Jittering: High variability but not moving far
    running_df['is_jittering'] = (running_df['roll_std'] > jitter_threshold) & (~running_df['is_running'])


def get_running_epochs(bsd,
                       behavior_type: str, # 'running' or 'stationary'
                       duration_threshold_in_s: float = 5.0,
                       running_speed_processing_args: dict = {
                            'mean_window_size': 2.0, # seconds
                            'std_window_size': 1.0, # seconds
                            'run_threshold': 5.0,  # speed threshold for running from rolling mean
                            'jitter_threshold': 0.5,  # std threshold for jittering from rolling std
                            },
                       ):
    ''' Get running or stationary epochs from BehaviorSessionDataset object
    Returns list of (start_time, end_time) tuples
    '''
    process_running_speed(bsd, **running_speed_processing_args)
    running_df = bsd.running_speed
    is_running = running_df['is_running'].values
    is_jittering = running_df['is_jittering'].values
    timestamps = running_df['timestamps'].values
    epochs = []
    in_epoch = False
    for i in range(1, len(is_running)):
        if behavior_type == 'running':
            condition = is_running[i]
        elif behavior_type == 'stationary':
            condition = not is_running[i] and not is_jittering[i]
        else:
            raise ValueError("behavior_type must be 'running' or 'stationary'")
        
        if condition and not in_epoch:
            start_time = timestamps[i]
            in_epoch = True
        elif not condition and in_epoch:
            end_time = timestamps[i]
            in_epoch = False
            if (end_time - start_time) >= duration_threshold_in_s:
                epochs.append((start_time, end_time))
    # Check if we ended while still in an epoch
    if in_epoch:
        end_time = timestamps[-1]
        if (end_time - start_time) >= duration_threshold_in_s:
            epochs.append((start_time, end_time))
    return epochs