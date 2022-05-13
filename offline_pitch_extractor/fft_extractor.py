from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks


def fft_pitch(data, sampling_frequency):

    # Get some useful statistics
    T = 1 / sampling_frequency  # Sampling period
    N = data.size  # Signal length in samples
    t = N / sampling_frequency  # Signal length in seconds

    Y_k = np.fft.fft(data)[0 : int(N / 2)] / N  # FFT
    Y_k[1:] = 2 * Y_k[1:]  # Single-sided spectrum
    Pxx = np.abs(Y_k)  # Power spectrum

    f = sampling_frequency * np.arange((N / 2)) / N  # frequencies

    auto = sm.tsa.acf(data, nlags=2000)
    peaks = find_peaks(auto)[0]  # Find peaks of the autocorrelation
    lag = peaks[0]  # Choose the first peak as our pitch component lag
    pitch = sampling_frequency / lag  # Transform lag into frequency

    # plotting

    fig, ax = plt.subplots()
    plt.plot(f[0:2000], Pxx[0:2000], linewidth=2)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.show()

    return pitch


"""
path = "../sounds/violin-B3.wav"
#path = "../sounds/sine-101.wav"

pitch = fft_pitch(path)
"""
