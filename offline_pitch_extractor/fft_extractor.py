import matplotlib.pyplot as plt
import numpy as np


def fft_extractor(audio, sr=44100):
    """
    Extracts the fundamental frequency given an input sound using the FFT method.
    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
    Returns:
        freq: the estimated fundamental frequency (float)
    """

    T = 1 / sr  # Sampling period
    N = audio.size  # Signal length in samples
    t = N / sr  # Signal length in seconds

    audio_fft = np.fft.fft(audio)[0 : int(N / 2)] / N  # FFT
    audio_fft[1:] = 2 * audio_fft[1:]  # Single-sided spectrum
    spec = np.abs(audio_fft)  # Power spectrum

    freqs = sr * np.arange((N / 2)) / N  # frequencies

    """th = 100
    idxs = [0]
    last_idx = -1

    for i in range(1, len(spec)):

        if last_idx > 0 and abs(last_idx - i) > 10:
            break

        elif spec[i] > th:
            last_idx = i
            idxs.append(i)

    freq = np.mean(freqs[idxs])"""
    # plotting
    fig, ax = plt.subplots()
    plt.plot(freqs[:2000], spec[:2000], linewidth=2)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.show()

    return freqs
