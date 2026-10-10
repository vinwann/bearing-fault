"""Utility snippets from assignment PDF pages 4–6.

Transcribed with indentation and wrapped lines restored; logic is unchanged.
"""

import numpy as np
from scipy import signal


# Unit / domain conversions
def convert_ms2_to_mms2(data: np.ndarray) -> np.ndarray:
    return data * 1000.0


def to_g(data: np.ndarray) -> np.ndarray:
    return data / 9.80665


def integrate_data(data: np.ndarray, sampling_rate: float) -> np.ndarray:
    return np.cumsum(data) / sampling_rate


# Filters
def bandpass_filter(
    data: np.ndarray,
    low_cutoff: int,
    high_cutoff: int,
    sampling_rate: int,
) -> np.ndarray:
    sos = signal.butter(
        6,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return np.asarray(signal.sosfilt(sos, data))


def highpass_filter(
    data: np.ndarray,
    low_cutoff: int,
    sampling_rate: int,
) -> np.ndarray:
    sos = signal.butter(
        6, low_cutoff, btype="highpass", fs=sampling_rate, output="sos"
    )
    return np.asarray(signal.sosfilt(sos, data))


# helpers
def remove_settling_time(
    data: np.ndarray,
    low_cutoff: int,
    original_length: int,
    sampling_rate: int,
) -> np.ndarray:
    fmax = sampling_rate / 2.56
    duration = (original_length / 2) / fmax
    multiplier = 2 if duration >= 2 else 1
    start = int(
        multiplier / low_cutoff * sampling_rate,
    )
    start = min(start, len(data) // 10)
    return data[start:]


def get_rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data)


def preprocess_acceleration(
    data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    length = len(data)
    low_cutoff = 10
    x = to_g(data)
    x = highpass_filter(x, low_cutoff, sampling_rate)
    x = signal.detrend(x, type="linear")
    x = remove_settling_time(x, low_cutoff, length, sampling_rate)
    x = remove_dc_offset(x)
    return x


def preprocess_velocity(
    data: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    length = len(data)
    low_cutoff = 10
    high_cutoff = 1000
    x = convert_ms2_to_mms2(data)
    x = integrate_data(x, sampling_rate)
    x = bandpass_filter(x, low_cutoff, high_cutoff, sampling_rate)
    x = signal.detrend(x, type="linear")
    x = remove_settling_time(x, low_cutoff, length, sampling_rate)
    x = remove_dc_offset(x)
    return x


def get_fft(
    data: np.ndarray,
    sampling_rate: int,
):
    N = len(data)
    line_of_resolution = N / 2.56
    maximum_frequency = sampling_rate / 2.56
    resolution = maximum_frequency / line_of_resolution
    nperseg = int(sampling_rate / resolution)
    nperseg = min(nperseg, len(data))
    noverlap = int(0.5 * nperseg)
    frequencies, psd = signal.welch(
        data,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        detrend=False,
    )
    mask = frequencies <= line_of_resolution
    frequencies = frequencies[mask]
    psd = psd[mask]
    df = frequencies[1] - frequencies[0]
    rms = np.sqrt(psd * df)
    return frequencies, rms


def get_features(
    data: np.ndarray,
    sampling_rate: int,
) -> dict:
    features = {}
    preprocessed_acceleration = preprocess_acceleration(data, sampling_rate)
    features["acceleration_rms"] = get_rms(preprocessed_acceleration)
    features["acceleration_fft"] = get_fft(preprocessed_acceleration, sampling_rate)
    preprocessed_velocity = preprocess_velocity(data, sampling_rate)
    features["velocity_rms"] = get_rms(preprocessed_velocity)
    features["velocity_fft"] = get_fft(preprocessed_velocity, sampling_rate)
    return features
