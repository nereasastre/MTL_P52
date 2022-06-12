import numpy as np
import scipy.fftpack
import util_functions as uf

fs = 44100  # sample frequency in Hz
window_size = 44100  # window size of the DFT in samples


# FFT pitch detector
def fft_pitch_detector(audio):
    magnitude_spec = abs(scipy.fftpack.fft(audio)[:len(audio) // 2])

    for i in range(int(62 / (fs / window_size))):
        magnitude_spec[i] = 0  # suppress mains hum

    max_index = np.argmax(magnitude_spec)
    pitch_detected = max_index * (fs / window_size)         # maximum frequency
    closest_note, closest_pitch, pitch_diff = uf.find_closest_note(pitch_detected)
    return pitch_detected, closest_note, closest_pitch, pitch_diff
