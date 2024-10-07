import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from typing import List, Tuple


# =========================== BLINKS DETECTION ===========================
# Helper function
def _interpolate_nans(array: np.ndarray, timestamps: np.ndarray, gap_dur: float = np.inf) -> np.ndarray:
    """
    Function finds sequences of NaN values, selects ones with
    duration <= 'gap_dur' and linearly interpolates them.
    :param array: array with NaNs.
    :param timestamps: timestamps of array.
    :param gap_dur: threshold gap duration.
    :return: array with interpolated gaps.
    """
    assert len(array.shape) == 1, "Only 1D array is allowed."
    assert array.shape == timestamps.shape, "'array' and 'timestamps' must correspond in shape."
    assert gap_dur > 0, "Gap duration must be positive."

    # Find index for nans where gaps are longer than 'gap_dur' samples
    mask = np.isnan(array)

    # If there are no nans, return
    if not np.any(mask):
        return array

    # Find onsets and offsets of gaps
    d = np.diff(np.concatenate((np.array([0]), mask * 1, np.array([0]))))
    onsets = np.where(d == 1)[0]
    offsets = np.where(d == -1)[0]

    # Decrease offsets come too late by -1
    if np.any(offsets >= len(array)):
        idx = np.where(offsets >= len(array))[0][0]
        offsets[idx] = offsets[idx] - 1

    dur = timestamps[offsets] - timestamps[onsets]

    # If the gaps are longer than 'gaps', replace temporarily with other values
    for i, on in enumerate(onsets):
        if dur[i] > gap_dur:
            array[onsets[i]:offsets[i]] = -1000

    # New is-nan mask after 'gaps' grasp
    mask = np.isnan(array)
    imask = ~mask
    array[mask] = np.interp(mask.nonzero()[0], imask.nonzero()[0], array[imask])

    # Put nans back
    array[array == -1000] = np.nan

    return array


# Helper function
def _nearest_odd_integer(x: float) -> int:
    """
    Method computes nearest odd integer to 'x'.
    """
    return int(2 * np.floor(np.abs(x) / 2) + 1) * np.sign(x)


# Helper function
def _mask2bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method constructs onset and offset arrays of indices based on mask.
    Onset - starting index of segment (segment is continuous block of ones),
    offset - ending index of blink.
    :param mask: boolean array.
    :return: [onsets, offsets].
    """

    # Find segments
    d = np.diff(np.hstack((0, mask, 0)))
    onsets = np.where(d == 1)[0]
    offsets = np.where(d == -1)[0] - 1

    # Match onsets with offsets
    if len(offsets) > len(onsets):
        if onsets[0] > offsets[0]:
            offsets = offsets[1:]
        else:
            offsets = offsets[:-1]
    elif len(offsets) < len(onsets):
        if onsets[0] > offsets[0]:
            onsets = onsets[1:]
        else:
            onsets = onsets[:-1]

    return onsets, offsets


# Helper function
def _merge_blinks(
        blink_onsets: List | np.ndarray,
        blink_offsets: List | np.ndarray,
        min_dur: int,
        min_separation: int,
        blink_properties: List | np.ndarray = None
) -> List[List]:
    """
    Method merges blinks given onsets and offsets, also collapses too short ones.
    :param blink_onsets: onsets of blinks, ms.
    :param blink_offsets: offsets of blinks, ms.
    :param min_dur: min duration of blink, ms.
    :param min_separation: min duration between blinks, ms.
    :param blink_properties: array of lists, properties of blinks.
    :return: array of triples (onset, offset, duration).
    """
    have_properties = blink_properties is not None

    # Merge blink candidate close together, and remove short, isolated ones
    new_onsets = []
    new_offsets = []
    new_properties = []
    change_onset = True
    temp_onset = None

    for i, onset in enumerate(blink_onsets):
        if change_onset:
            temp_onset = blink_onsets[i]

        if i < len(blink_onsets) - 1:
            if (blink_onsets[i + 1] - blink_offsets[i]) < min_separation:

                change_onset = False
            else:
                change_onset = True

                # Remove blink with too short duration
                if (blink_offsets[i] - temp_onset) < min_dur:
                    continue

                new_onsets.append(temp_onset)
                new_offsets.append(blink_offsets[i])
                if have_properties:
                    new_properties.append(blink_properties[i, :])
        else:

            # Remove blink with too short duration
            if (blink_offsets[i] - temp_onset) < min_dur:
                continue

            new_onsets.append(temp_onset)
            new_offsets.append(blink_offsets[i])
            if have_properties:
                new_properties.append(blink_properties[i, :])

    # Compute durations and convert to array
    blinks = []
    for i in range(len(new_onsets)):
        dur = new_offsets[i] - new_onsets[i]
        blink_data = [new_onsets[i], new_offsets[i], dur]
        if have_properties:
            blink_data += list(new_properties[i])
        blinks.append(blink_data)

    return blinks


# Helper function
def _indices_to_values(
        onsets: np.ndarray,
        offsets: np.ndarray,
        timestamps: np.ndarray,
        Fs: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method converts index-based onsets/offsets to
    timestamp-based onsets/offsets.
    :param onsets: indexes of starting blink indexes.
    :param offsets: indexes of ending blink indexes.
    :param timestamps: data recording timestamps.
    :param Fs: sample rate of eye tracker, Hz.
    :return: onsets, offsets.
    """

    # Convert onsets/offsets to ms
    blinks = []
    for onset, offset in zip(onsets, offsets):
        blinks.append([timestamps[onset], timestamps[offset]])

    if Fs is not None:
        assert Fs > 0
        # Remove blinks with on-, or offsets that happened in a period of missing data
        # (i.e., where samples are completely lost, for some reason)
        idx = np.where(np.diff(timestamps) > (2 * 1 / Fs * 1000))  # Missing data where deltaT > 2 * 1/Fs

        for i, blink in enumerate(blinks):
            for idx_k in idx[0]:
                if np.logical_and(blink[0] >= timestamps[idx_k], blink[0] <= timestamps[idx_k + 1]) or \
                        np.logical_and(blink[1] >= timestamps[idx_k], blink[1] <= timestamps[idx_k + 1]):
                    blinks.pop(i)
                    break

    onsets = np.array([b[0] for b in blinks])
    offsets = np.array([b[1] for b in blinks])

    return onsets, offsets

# Helper function
def _apply_moving_average(
        pupil_signal: np.ndarray,
        timestamps: np.ndarray,
        is_na: np.ndarray,
        max_window_size: float
) -> np.ndarray:
    """
    Method applies moving average for pupillometry data. Nan values are remained.
    :param pupil_signal: sizes of pupil.
    :param timestamps: data recording timestamps.
    :param is_na: boolean array representing whether pupil is nan or not.
    :param max_window_size: maximum size of smoothing window, in milliseconds.
    :return: smoothed pupil sizes.
    """

    if len(pupil_signal) <= 2:
        return pupil_signal
    assert (len(pupil_signal) == len(timestamps) == len(is_na))
    assert (max_window_size > 0.0)

    n = len(pupil_signal)
    tmp = pupil_signal.copy()
    tmp = np.where(is_na, 0, tmp)
    csum_size = np.cumsum(tmp)
    csum_time = timestamps - np.roll(timestamps, shift=1)
    csum_time[0] = 0
    csum_time = np.cumsum(csum_time)

    smoothed_sizes = np.zeros((n,))
    smoothed_sizes[0] = pupil_signal[0]

    window_end = -1
    for i in range(1, n):
        if is_na[i]:
            window_end = -1
            smoothed_sizes[i] = np.nan
            continue

        window_end = max([window_end - 1, i + 1])
        while (window_end < n and
               not is_na[window_end] and
               (csum_time[window_end] - csum_time[i - 1]) <= max_window_size):
            window_end += 1
        smoothed_sizes[i] = (csum_size[window_end - 1] - csum_size[i - 1]) / (window_end - i)

    return smoothed_sizes


# Blinks Detector
def detect_blinks_pupil_missing(
        pupil_signal: np.ndarray,
        timestamps: np.ndarray,
        min_separation: int = 100,
        min_dur: int = 20,
        smoothing_window_size: int = 10,  # 10ms
        return_mask: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """
    Method detects blinks based on size of pupil and missing recordings (NaN) in its data.
    Result is boolean array of the same
    length as pupil_signal, with 1 indicating blink, 0 - not blink.
    Taken from [PyTrack paper](https://link.springer.com/article/10.3758/s13428-020-01392-6):
    :param pupil_signal: size of right or left pupil.
    :param timestamps: data recording timestamps.
    :param smoothing_window_size: maximum size of smoothing window, ms.
    :param min_separation: min time interval between detected blinks, ms.
    :param min_dur: min duration of blink, ms.
    :param return_mask: bool = False
    :return: detected blinks dataframe.
    """

    assert not np.isnan(timestamps).any(), "Timestamps must not be nan."
    assert len(pupil_signal) > 10, "There must be at least 10 recordings."

    # get mask for NaN values
    is_na = np.isnan(pupil_signal).astype(np.int32)
    assert np.sum(is_na) > 0, ("This algorithm is based on missing values in pupil size recordings data, but"
                               "provided 'pupil_signal' array does not contain NaNs.")

    # smooth using moving average filter
    smoothed_pupil_signal = _apply_moving_average(pupil_signal, timestamps, is_na, smoothing_window_size)

    # calculate difference between neighbor values
    diff = smoothed_pupil_signal - np.roll(smoothed_pupil_signal, shift=1)
    diff = np.where(np.roll(is_na, shift=1), smoothed_pupil_signal, diff)
    diff[0] = 0

    # naive approach - missing value is blink
    is_blink = is_na
    n = len(pupil_signal)

    # improve approach - expand blink boundaries using monotonic sequences
    for i in range(1, n - 1):
        if is_blink[i]:
            continue

        # monotonic pupil size decrease
        if is_blink[i + 1]:
            left = i
            assert not np.isnan(diff[left])
            while left >= 0 and not is_blink[left] and diff[left] < 0:
                is_blink[left] = 1
                left -= 1

        # monotonic pupil size increase
        if is_blink[i - 1]:
            right = i
            while right < n and not is_blink[right] and diff[right] > 0:
                is_blink[right] = 1
                right += 1

    onsets, offsets = _mask2bounds(is_blink)
    onsets, offsets = _indices_to_values(onsets, offsets, timestamps)

    # Merge blinks closer than x ms, and remove short blinks
    blinks = _merge_blinks(onsets, offsets, min_dur, min_separation)
    df = pd.DataFrame(blinks,
                      columns=['onset', 'offset', 'duration'])
    if return_mask:
        return df, is_blink
    return df


# Blinks Detector
def detect_blinks_pupil_vt(
    pupil_signal: np.ndarray,
    timestamps: np.ndarray,
    Fs: int,
    gap_dur: int = 20,
    min_dur: int = 20,
    remove_outliers: bool = False,
    window_len: int = None,
    min_pupil_size: int = 2,
    min_separation: int = 50,
    return_mask: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """
    Method detects blinks based on pupil sizes and change of
    pupil sizes. Taken from https://link.springer.com/article/10.3758/s13428-023-02333-9,
    code taken from https://github.com/marcus-nystrom/BlinkDetector.
    :param pupil_signal: size of right of left pupil, mm.
    :param timestamps: data recording timestamps, ms.
    :param Fs: sample rate of eye tracker, Hz.
    :param gap_dur: max gaps between periods of data loss, ms.
    :param min_dur: min duration of blink, ms.
    :param remove_outliers: whether to remove outliers.
    :param window_len: size of window to use if 'remove_outliers' = True.
    :param min_pupil_size: min value of pupil size considered to be recorded correctly, mm.
    :param min_separation: min time interval between detected blinks, ms.
    :param return_mask: if True, then mask showing blink classification is also returned.
    :return: detected blinks dataframe.
    """

    # Remove outliers
    if remove_outliers:
        if np.isnan(window_len):
            window_len_samples = len(pupil_signal)
        else:
            window_len_samples = int(
                Fs / 1000 * window_len)  # in ms window over which to exclude outliers

        ps = pupil_signal.copy()
        ps[ps < min_pupil_size] = np.nan

        for k in np.arange(len(ps) - window_len_samples + 1):
            temp = pupil_signal[k: (k + window_len_samples)].copy()

            if len(temp) == 0:
                continue

            m = np.nanmean(temp)
            sd = np.nanstd(temp)
            idx = (temp > (m + 3 * sd)) | (temp < (m - 3 * sd))
            temp[idx] = np.nan
            ps[k: (k + window_len_samples)] = temp

        pupil_signal = ps

    # Interpolate short periods of data loss
    pupil_signal = _interpolate_nans(
        pupil_signal,
        timestamps,
        gap_dur=gap_dur
    )

    # Convert to bounds and clean up
    is_blink = np.isnan(pupil_signal) * 1
    onsets, offsets = _mask2bounds(is_blink)
    onsets, offsets = _indices_to_values(onsets, offsets, timestamps, Fs=Fs)

    # Merge blinks closer than x ms, and remove short blinks
    blinks = _merge_blinks(onsets, offsets, min_dur, min_separation)
    df = pd.DataFrame(blinks,
                      columns=['onset', 'offset', 'duration'])
    if return_mask:
        return df, is_blink
    return df


# Blinks Detector
def detect_blinks_eo(
        eye_openness_signal: np.ndarray,
        timestamps: np.ndarray,
        Fs: int,
        gap_dur: int = 30,
        filter_length: int = 25,
        min_blink_length: int = 15,
        min_amplitude: int = 0.1,
        min_separation: int = 100,
        return_eo_vel: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, np.ndarray]:
    """
    Method detects blinks based on Eye Openness (EO) signal.
    Taken from https://link.springer.com/article/10.3758/s13428-023-02333-9,
    code taken from https://github.com/marcus-nystrom/BlinkDetector.
    :param eye_openness_signal: eye openness signal.
    :param timestamps: data recording timestamps, ms.
    :param Fs: sample rate of eye tracker, Hz.
    :param gap_dur: max gaps between periods of data loss, ms.
    :param filter_length: length of Savitzky-Golay filter, ms.
    :param min_blink_length: min length of detected blinks, ms.
    :param min_separation: min time interval between detected blinks, ms.
    :param min_amplitude: fraction of fully opened eye.
    :param return_eo_vel: if True, then computed velocity of EO signal is returned.
    :return: detected blinks dataframe.
    """

    ms_to_sample = Fs / 1000
    sample_to_ms = 1000 / Fs

    # Assumes the eye is mostly open during the trial
    fully_open = np.nanmedian(eye_openness_signal, axis=0)
    min_amplitude = fully_open * min_amplitude  # Equivalent to height in 'find_peaks'

    # detection parameters in samples
    distance_between_blinks = 1
    min_blink_length = min_blink_length * ms_to_sample
    filter_length = _nearest_odd_integer(filter_length * ms_to_sample)

    # Interpolate gaps
    eye_openness_signal = _interpolate_nans(eye_openness_signal, timestamps,
                                            gap_dur=int(gap_dur))

    # Filter eyelid signal and compute
    eye_openness_signal_filtered = savgol_filter(
        eye_openness_signal, filter_length, 2,
        mode='nearest'
    )
    eye_openness_signal_vel = savgol_filter(
        eye_openness_signal, filter_length, 2,
        deriv=1,  mode='nearest'
    ) * Fs

    # Velocity threshold for on-, and offsets
    T_vel = stats.median_abs_deviation(eye_openness_signal_vel, nan_policy='omit') * 3

    # Turn blink signal into something that looks more like a saccade signal
    eye_openness_signal_inverse = (eye_openness_signal_filtered -
                                   np.nanmax(eye_openness_signal_filtered)) * -1
    peaks, properties = find_peaks(eye_openness_signal_inverse, height=None,
                                   distance=distance_between_blinks,
                                   width=min_blink_length)

    # Filter out not so 'prominent peaks'
    """
    The prominence of a peak may be defined as the least drop in height
     necessary in order to get from the summit [peak] to any higher terrain.
    """
    idx = properties['prominences'] > min_amplitude
    peaks = peaks[idx]
    for key in properties.keys():
        properties[key] = properties[key][idx]

    # Find peak opening/closing velocity by searching for max values
    # within a window from the peak
    blink_properties = []
    for i, peak_idx in enumerate(peaks):

        # Width of peak
        width = properties['widths'][i]

        ### Compute opening/closing velocity
        # First eye opening velocity (when eyelid opens after a blink)
        peak_right_idx = np.nanargmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])
        peak_right_idx = np.nanmin([peak_right_idx, len(eye_openness_signal_vel)])
        idx_max_opening_vel = int(peak_idx + peak_right_idx)
        time_max_opening_vel = timestamps[idx_max_opening_vel]
        opening_velocity = np.nanmax(eye_openness_signal_vel[peak_idx:int(peak_idx + width)])

        # Then eye closing velocity (when eyelid closes in the beginning of a blink)
        peak_left_idx = width - np.nanargmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx]) + 1
        peak_left_idx = np.nanmax([peak_left_idx, 0])
        idx_max_closing_vel = int(peak_idx - peak_left_idx + 1)
        time_max_closing_vel = timestamps[idx_max_closing_vel]
        closing_velocity = np.nanmin(eye_openness_signal_vel[np.max([0, int(peak_idx - width)]):peak_idx])

        # Identify on and offsets (go from peak velocity backward/forward)
        temp = eye_openness_signal_vel[idx_max_opening_vel:]
        if np.any(temp <= (T_vel / 3)):
            offset = np.where(temp <= (T_vel / 3))[0][0]
        else:
            offset = len(temp)

        # make sure the blink period stop when encountering nan-data
        # If it does, make the opening phase parameters invalid
        if np.any(np.isnan(temp)):
            offset_nan = np.where(np.isnan(temp))[0][0]
            offset = np.min([offset, offset_nan])

        offset_idx = int(idx_max_opening_vel + offset - 1)

        temp = np.flip(eye_openness_signal_vel[:idx_max_closing_vel])
        if np.any(temp >= -T_vel):
            onset = np.where(temp >= -T_vel)[0][0]
        else:
            onset = 0

        if np.any(np.isnan(temp)):
            onset_nan = np.where(np.isnan(temp))[0][0]
            onset = np.min([onset, onset_nan])

        onset_idx = int(idx_max_closing_vel - onset)


        # Compute openness at onset, peak, and offset
        openness_at_onset = eye_openness_signal_filtered[onset_idx]
        openness_at_offset = eye_openness_signal_filtered[offset_idx]
        openness_at_peak = eye_openness_signal_filtered[peak_idx]

        # Compute amplitudes for closing and opening phases
        closing_amplitude = np.abs(openness_at_onset - openness_at_peak)
        opening_amplitude = np.abs(openness_at_offset - openness_at_peak)

        distance_onset_peak_vel = np.abs(eye_openness_signal_filtered[onset_idx] -
                                         eye_openness_signal_filtered[idx_max_closing_vel]) # mm
        timediff_onset_peak_vel = np.abs(onset_idx - idx_max_closing_vel) * sample_to_ms # ms

        # Onset and peak cannot be too close in space and time
        if (distance_onset_peak_vel < 0.1) or (timediff_onset_peak_vel < 10):
            continue

        if np.min([opening_velocity, np.abs(closing_velocity)]) < (T_vel * 2):
            continue

        blink_properties.append([timestamps[onset_idx],
                                 timestamps[offset_idx],
                                 timestamps[offset_idx] - timestamps[onset_idx],
                                 timestamps[peak_idx],
                                 openness_at_onset, openness_at_offset,
                                 openness_at_peak,
                                 time_max_opening_vel,
                                 time_max_closing_vel,
                                 opening_velocity, closing_velocity,
                                 opening_amplitude, closing_amplitude])

    # Are there any blinks found?
    if len(blink_properties) == 0:
        bp = []
    else:

        # Merge blinks too close together in time
        blink_temp = np.array(blink_properties)
        blink_onsets = blink_temp[:, 0]
        blink_offsets = blink_temp[:, 1]

        bp = _merge_blinks(
            blink_onsets, blink_offsets, int(min_blink_length), min_separation,
            blink_properties=blink_temp[:, 3:])

    # Convert to dataframe
    df = pd.DataFrame(bp,
                      columns=['onset', 'offset', 'duration',
                               'time_peak',
                               'openness_at_onset',
                               'openness_at_offset',
                               'openness_at_peak',
                               'time_peak_opening_velocity',
                               'time_peak_closing_velocity',
                               'peak_opening_velocity',
                               'peak_closing_velocity',
                               'opening_amplitude',
                               'closing_amplitude'])

    idx = df.index[df['openness_at_peak'] < 0]
    df.loc[idx, 'openness_at_peak'] = 0
    if return_eo_vel:
        return df, eye_openness_signal_vel
    return df


if __name__ == "__main__":
    from numpy import array, nan

    test_pupil_sizes = array([1.2 , 1.12, 1.15, 1.3 , 1.21, 1.25, 0.9 ,  nan,  nan,  nan,  nan, 0.98,
       0.95, 1.2 , 1.33, 1.54, 1.3 , 1.3 , 1.25, 1.44])
    test_timestamps = array([487383.208, 487391.543, 487399.885, 487408.216, 487416.551,
       487424.872, 487433.203, 487441.537, 487449.87 , 487458.203,
       487466.532, 487474.868, 487483.206, 487491.528, 487499.912,
       487508.225, 487516.531, 487524.865, 487533.192, 487540.733])

    # print(detect_blinks_pupil_missing(test_pupil_sizes, test_timestamps))
    #
    # print(detect_blinks_pupil_vt(test_pupil_sizes, test_timestamps, Fs=120))

    print(detect_blinks_eo(test_pupil_sizes, test_timestamps, Fs=120)[0])
