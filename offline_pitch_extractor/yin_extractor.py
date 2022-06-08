import numpy as np
import matplotlib.pyplot as plt
from numpy import compat
from scipy.signal import fftconvolve


def difference_function_original(x, N, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    Original algorithm.

    Args:
        x: audio data
        N: length of data
        tau_max: integration window size

    Returns:
        difference function (list of float)
    """

    df = [0] * tau_max
    for tau in range(1, tau_max):
         for j in range(0, N - tau_max):
             tmp = compat.long(x[j] - x[j + tau])
             df[tau] += tmp * tmp
    return df


def difference_function_scipy(x, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    Faster implementation of the difference function.
    The required calculation can be easily evaluated by Autocorrelation function or similarly by convolution.
    Wiener–Khinchin theorem allows computing the autocorrelation with two Fast Fourier transforms (FFT), with time complexity O(n log n).
    This function use an accelerated convolution function fftconvolve from Scipy package.

    Args:
        x: audio data
        tau_max: integration window size

    Returns:
        difference function (list of float)
    """
    x = np.array(x, np.float64)
    w = x.size
    x_cumsum = np.concatenate((np.array([0]), (x * x).cumsum()))
    conv = fftconvolve(x, x[::-1])
    tmp = x_cumsum[w:0:-1] + x_cumsum[w] - x_cumsum[:w] - 2 * conv[w - 1:]
    return tmp[:tau_max + 1]


def difference_function(x, tau_max):
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    Fastest implementation. Use the same approach than differenceFunction_scipy.
    This solution is implemented directly with Numpy fft.

    Args:
        x: audio data (list of float)
        tau_max: integration window size (float)

    Returns:
        difference function (list of float)

    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulative_mean_normalized_difference_function(df, N):
    """
    Compute cumulative mean normalized difference function (CMND).
    This corresponds to equation (8) in [1]
    Args:
        df: Difference function (list of float)
        N: length of data (int)

    Returns:
        cumulative mean normalized difference function (list of float)
    """

    cmndf = df[1:] * range(1, N) / np.cumsum(df[1:]).astype(float) #scipy method
    return np.insert(cmndf, 0, 1)


def get_pitch(cmdf, tau_min, tau_max, harm_th=0.1):
    """
    Return fundamental period of a frame based on CMND function.
    Args:
        cmdf: Cumulative Mean Normalized Difference function
        tau_min: minimum period for speech
        tau_max: maximum period for speech
        harm_th: harmonicity threshold to determine if it is necessary to compute pitch frequency

    Returns:
        tau: fundamental period if there is values under threshold, 0 otherwise (float)
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harm_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0    # if unvoiced


def compute_yin(sig, sr, w_len=512, w_step=256, f0_min=100.0, f0_max=500.0, harm_thresh=0.1):
    """
    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
    Args:
        sig: Audio signal (list of float)
        sr: sampling rate (int)
        w_len: size of the analysis window (samples)
        w_step: size of the lag between two consecutives windows (samples)
        f0_min: Minimum fundamental frequency that can be detected (hertz)
        f0_max: Maximum fundamental frequency that can be detected (hertz)
        harm_thresh: Threshold of detection. # todo review of: The yalgorithmù return the first minimum of the CMND fubction below this threshold.

    Returns:
        pitches: list of fundamental frequencies,
        harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
        argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
        times: list of time of each estimation
    """

    print('Yin: compute yin algorithm')
    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    time_scale = range(0, len(sig) - w_len, w_step)  # time values for each analysis window
    times = [t/float(sr) for t in time_scale]
    frames = [sig[t:t + w_len] for t in time_scale]

    pitches = [0.0] * len(time_scale)
    harmonic_rates = [0.0] * len(time_scale)
    argmins = [0.0] * len(time_scale)

    for i, frame in enumerate(frames):

        # Compute YIN
        df = difference_function(frame, tau_max)
        cmdf = cumulative_mean_normalized_difference_function(df, tau_max)
        p = get_pitch(cmdf, tau_min, tau_max, harm_thresh)

        # Get results
        if np.argmin(cmdf)>tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0: # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else: # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)

    return pitches, harmonic_rates, argmins, times


def yin_extractor(audio, sr=44100, w_len=1024, w_step=256, f0_min=70.0, f0_max=1500.0, harm_thresh=0.07):
    """
    Run the computation of the Yin algorithm on a example file.
    Write the results (pitches, harmonic rates, parameters ) in a numpy file.

    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
        w_len: length of the window (int)
        w_step: length of the "hop" size (int)
        f0_min: minimum f0 in Hertz (float)
        f0_max: maximum f0 in Hertz (float)
        harm_thresh: harmonic threshold (float)
    Returns:
        freq: the estimated fundamental frequency (float)
    """
    freqs, harmonic_rates, argmins, times = compute_yin(audio, sr, w_len, w_step, f0_min, f0_max, harm_thresh)

    # freq = np.mean(freqs)
    return freqs


def yin_plot(audio, sr = 44100, w_len=1024, w_step=256, f0_min=70.0, f0_max=200.0, harm_thresh=0.85):
    """
    plot the results (pitches, harmonic rates, parameters )
    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
        w_len: length of the window (int)
        w_step: length of the "hop" size (int)
        f0_min: minimum f0 in Hertz (float)
        f0_max: maximum f0 in Hertz (float)
        harm_thresh: harmonic threshold (float)
    """

    pitches, harmonic_rates, argmins, times = compute_yin(audio, sr, w_len, w_step, f0_min, f0_max, harm_thresh)

    duration = len(audio)/float(sr)

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot([float(x) * duration / len(audio) for x in range(0, len(audio))], audio)
    ax1.set_title('Audio data')
    ax1.set_ylabel('Amplitude')
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot([float(x) * duration / len(pitches) for x in range(0, len(pitches))], pitches)
    ax2.set_title('F0')
    ax2.set_ylabel('Frequency (Hz)')
    ax3 = plt.subplot(4, 1, 3, sharex=ax2)
    ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], harmonic_rates)
    ax3.plot([float(x) * duration / len(harmonic_rates) for x in range(0, len(harmonic_rates))], [harm_thresh] * len(harmonic_rates), 'r')
    ax3.set_title('Harmonic rate')
    ax3.set_ylabel('Rate')
    ax4 = plt.subplot(4, 1, 4, sharex=ax2)
    ax4.plot([float(x) * duration / len(argmins) for x in range(0, len(argmins))], argmins)
    ax4.set_title('Index of minimums of CMND')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (seconds)')
    plt.show()