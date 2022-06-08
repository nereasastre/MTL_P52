import numpy as np
import scipy.fftpack

fs = 44100  # sample frequency in Hz
window_size = 44100  # window size of the DFT in samples


# FFT pitch detector
def fft_pitch_detector(window_samples):
    magnitude_spec = abs(scipy.fftpack.fft(window_samples)[:len(window_samples) // 2])

    for i in range(int(62 / (fs / window_size))):
        magnitude_spec[i] = 0  # suppress mains hum

    max_index = np.argmax(magnitude_spec)
    pitch_detected = max_index * (fs / window_size)         # maximum frequency
    return pitch_detected
