import numpy as np
import os
import sounddevice as sd
import time
from django.conf import settings

#import hps_extractor as hpse
from real_time_pitch_extractor.hps_extractor import hps_pitch_detector
# General settings that can be changed by the user
sr = 44100  # sample frequency in Hz
window_size = 44100  # window size of the DFT in samples
window_step = 12000  # step size of window


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
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))  # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]  # remove old samples

        pitch_detected, closest_pitch, closest_note, pitch_diff = hps_pitch_detector(
            callback.window_samples)  # extract pitch
        pitch_diff = round(pitch_diff, 1)
        callback.noteBuffer.insert(0, closest_note)  # note that this is a ringbuffer
        callback.noteBuffer.pop()

        settings.PITCH_DETECTED = pitch_detected
        settings.CLOSEST_NOTE = closest_note
        settings.CLOSEST_PITCH = closest_pitch
        settings.PITCH_DIFF = pitch_diff

        os.system('cls' if os.name == 'nt' else 'clear')
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            print(
                f"Pitch detected: {pitch_detected} --> Closest note: {closest_note} ({closest_pitch}) --> Pitch difference: {pitch_diff}")
        else:
            print(f"Closest note: ...")

    else:
        print('no input')


# Start the microphone input stream

def real_time(scallback):
    try:
        print("Starting HPS guitar tuner...")
        with sd.InputStream(channels=1, callback=callback, blocksize=window_step, samplerate=sr):
            while settings.RECORD:
                scallback()

    except Exception as e:
        print(str(e))
