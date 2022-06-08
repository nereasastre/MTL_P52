import numpy as np
import scipy.fftpack
import os
import copy

fs = 48000  # sample frequency in Hz
window_size = 48000  # window size of the DFT in samples

num_hps = 5  # max number of harmonic product spectrums
power_th = 1e-6  # tuning is activated if the signal power exceeds this threshold
white_noise_th = 0.2  # everything under white_noise_th*avg_energy_per_freq is cut off

delta_freq = fs / window_size  # frequency step width of the interpolated DFT
octave_bands = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

hanning_window = np.hanning(window_size)

# Harmonic Product Spectrum (HPS) pitch detector


def hps_pitch_detector(window_samples):
    # skip if signal power is too low
    signal_power = (np.linalg.norm(window_samples, ord=2) ** 2) / len(window_samples)
    if signal_power < power_th:
        os.system("cls" if os.name == "nt" else "clear")
        print("Closest note: ...")

    # avoid spectral leakage by multiplying the signal with a hanning window
    hanning_samples = window_samples * hanning_window
    magnitude_spec = abs(
        scipy.fftpack.fft(hanning_samples)[: len(hanning_samples) // 2]
    )

    # supress mains hum, set everything below 62Hz to zero
    for i in range(int(62 / delta_freq)):
        magnitude_spec[i] = 0

    # calculate average energy per frequency for the octave bands and suppress everything below it
    for j in range(len(octave_bands) - 1):
        ind_start = int(octave_bands[j] / delta_freq)
        ind_end = int(octave_bands[j + 1] / delta_freq)
        ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
        avg_energy_per_freq = (
            np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2
        ) / (ind_end - ind_start)
        avg_energy_per_freq = avg_energy_per_freq**0.5
        for i in range(ind_start, ind_end):
            magnitude_spec[i] = (
                magnitude_spec[i]
                if magnitude_spec[i] > white_noise_th * avg_energy_per_freq
                else 0
            )

    # interpolate spectrum
    mag_spec_ipol = np.interp(
        np.arange(0, len(magnitude_spec), 1 / num_hps),
        np.arange(0, len(magnitude_spec)),
        magnitude_spec,
    )
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculate the HPS
    for i in range(num_hps):
        tmp_hps_spec = np.multiply(
            hps_spec[: int(np.ceil(len(mag_spec_ipol) / (i + 1)))],
            mag_spec_ipol[:: (i + 1)],
        )
        if not any(tmp_hps_spec):
            break
        hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    pitch_detected = max_ind * (fs / window_size) / num_hps

    return pitch_detected
