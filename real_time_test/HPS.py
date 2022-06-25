"""
Guitar tuner script based on the Harmonic Product Spectrum (HPS)

MIT License
Copyright (c) 2021 chciken
"""

import copy
import os
import time
import numpy as np
import scipy.fftpack
import sounddevice as sd

# General settings that can be changed by the user
fs = 48000  # sample frequency in Hz
window_size = 48000  # window size of the DFT in samples
window_step = 12000  # step size of window
num_hps = 5  # max number of harmonic product spectrums
power_th = 1e-6  # tuning is activated if the signal power exceeds this threshold
concert_pitch = 440  # defining a1
white_noise_th = 0.2  # everything under white_noise_th*avg_energy_per_freq is cut off

window_size_sec = window_size / fs  # length of the window in seconds
Ts = 1 / fs  # length between two samples in seconds
delta_freq = fs / window_size  # frequency step width of the interpolated DFT
octave_bands = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]


hanning_window = np.hanning(window_size)

CONCERT_PITCH = 440
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch_detected):
    i = int(np.round(np.log2(pitch_detected / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    closest_pitch = round(closest_pitch, 2)
    diff = pitch_detected - closest_pitch
    diff = round(diff, 2)
    return closest_note, closest_pitch, diff


def callback(indata, frames, time, status):
    """
    Callback function of the InputStream method.
    That's where the magic happens ;)
    """
    # define static variables
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(window_size)]
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]

    if status:
        print(status)
        return
    if any(indata):
        callback.window_samples = np.concatenate(
            (callback.window_samples, indata[:, 0])
        )  # append new samples
        callback.window_samples = callback.window_samples[
            len(indata[:, 0]) :
        ]  # remove old samples

        # skip if signal power is too low
        signal_power = (np.linalg.norm(callback.window_samples, ord=2) ** 2) / len(
            callback.window_samples
        )
        if signal_power < power_th:
            os.system("cls" if os.name == "nt" else "clear")
            print("Closest note: ...")
            return
        # avoid spectral leakage by multiplying the signal with a hanning window
        hanning_samples = callback.window_samples * hanning_window
        magnitude_spec = abs(
            scipy.fftpack.fft(hanning_samples)[: len(hanning_samples) // 2]
        )

        # supress mains hum, set everything below 62Hz to zero
        for i in range(int(62 / delta_freq)):
            magnitude_spec[i] = 0

        # calculate average energy per frequency for the octave bands
        # and suppress everything below it
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
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(
            mag_spec_ipol, ord=2
        )  # normalize it

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

        closest_pitch, closest_note, pitch_diff = find_closest_note(pitch_detected)
        pitch_diff = round(pitch_diff, 1)
        callback.noteBuffer.insert(0, closest_note)  # note that this is a ringbuffer
        callback.noteBuffer.pop()

        os.system("cls" if os.name == "nt" else "clear")
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(
            callback.noteBuffer
        ):
            print(
                f"Pitch detected: {pitch_detected} --> Closest note: {closest_note} ({closest_pitch}) --> Pitch difference: {pitch_diff}"
            )
        else:
            print(f"Closest note: ...")

    else:
        print("no input")


# Start the microphone input stream

try:
    print("Starting HPS guitar tuner...")
    with sd.InputStream(
        channels=1, callback=callback, blocksize=window_step, samplerate=fs
    ):
        while True:
            time.sleep(0.5)
except Exception as e:
    print(str(e))
