import sounddevice as sd
import numpy as np
import os
import fft_extractor as ffte
import util_functions as uf
import yin_extractor as ye
import auto_extractor as ae
import crepe_extractor as ce
import zero_cross_extractor as zce

# General settings
fs = 44100  # sample frequency in Hz
window_size = 44100  # window size of the DFT in samples
window_step = 21050  # step size of window


# The sounddecive callback function
# Provides us with new data once window_step samples have been fetched


def callback(indata, frames, time, status):
    window_samples = [0 for _ in range(window_size)]
    window_samples = np.concatenate((window_samples, indata[:, 0]))  # append new samples
    window_samples = window_samples[len(indata[:, 0]):]  # remove old samples
    if status:
        print(status)
    if any(indata):
        # Call fft detection function to obtain pitch detected and its closest note. We can use other pitch detectors.
        # pitch_detected = ffte.fft_pitch_detector(window_samples)
        pitch_detected = ffte.fft_pitch_detector(window_samples)
        closest_note, closest_pitch, pitch_diff = uf.find_closest_note(pitch_detected)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Pitch detected: {pitch_detected} --> Closest note: {closest_note} ({closest_pitch}) --> Pitch difference: {pitch_diff}")
    else:
        print('no input')


# Start the microphone input stream

try:
    with sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=fs):
        while True:
            pass
except Exception as e:
    print(str(e))
