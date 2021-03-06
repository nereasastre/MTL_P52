import sounddevice as sd
import numpy as np
import scipy.fftpack
import os

# General settings
fs = 44100  # sample frequency in Hz
window_size = 44100  # window size of the DFT in samples
window_step = 21050  # step size of window
window_size_sec = window_size / fs  # length of the window in seconds
Ts = 1 / fs  # length between two samples in seconds
window_samples = [0 for _ in range(window_size)]

# This function finds the closest note for a given pitch
concert_pitch = 440
all_notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch / concert_pitch) * 12))
    closest_note = all_notes[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = concert_pitch * 2 ** (i / 12)
    return closest_note, closest_pitch


# The sounddecive callback function
# Provides us with new data once window_step samples have been fetched


def callback(indata, frames, time, status):
    global window_samples
    if status:
        print(status)
    if any(indata):
        window_samples = np.concatenate(
            (window_samples, indata[:, 0])
        )  # append new samples
        window_samples = window_samples[len(indata[:, 0]) :]  # remove old samples
        magnitude_spec = abs(
            scipy.fftpack.fft(window_samples)[: len(window_samples) // 2]
        )

        for i in range(int(62 / (fs / window_size))):
            magnitude_spec[i] = 0  # suppress mains hum

        max_index = np.argmax(magnitude_spec)
        max_freq = max_index * (fs / window_size)
        closest_note, closest_pitch = find_closest_note(max_freq)

        os.system("cls" if os.name == "nt" else "clear")
        print(f"Closest note: {closest_note} {max_freq:.1f}/{closest_pitch:.1f}")
    else:
        print("no input")


# Start the microphone input stream


try:
    with sd.InputStream(
        channels=1, callback=callback, blocksize=window_step, samplerate=fs
    ):
        while True:
            pass
except Exception as e:
    print(str(e))
