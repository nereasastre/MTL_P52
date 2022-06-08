import statsmodels.api as sm
from scipy.signal import find_peaks


def auto_extractor(audio, sr=44100):
    """
    Extracts the fundamental frequency given an input sound using the FFT method.
    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
    Returns:
        freq: the estimated fundamental frequency (float)
    """
    # Get some useful statistics
    auto = sm.tsa.acf(audio, nlags=2000)
    peaks = find_peaks(auto)[0]  # Find peaks of the autocorrelation
    lag = peaks[0]  # Choose the first peak as our pitch component lag
    freq = sr / lag  # Transform lag into frequency

    return freq
