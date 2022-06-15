import os
from time import sleep

import numpy as np
import sounddevice as sd
from django.conf import settings

# General settings
from real_time_pitch_extractor.fft_extractor import fft_pitch_detector

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

        pitch_detected, closest_note, closest_pitch, pitch_diff = fft_pitch_detector(window_samples)
        settings.PITCH_DETECTED = pitch_detected
        settings.CLOSEST_NOTE = closest_note
        settings.CLOSEST_PITCH = closest_pitch
        settings.PITCH_DIFF = pitch_diff

        os.system('cls' if os.name == 'nt' else 'clear')
        #print(f"Pitch detected: {pitch_detected} --> Closest note: {closest_note} ({closest_pitch}) --> Pitch difference: {pitch_diff}")
    else:
        print('no input')


def real_time(scallback):
    # Start the microphone input stream
    try:
        with sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=fs):
            while settings.RECORD:
                scallback()
                sleep(0.5)
    except Exception as e:
        print(str(e))
        return
