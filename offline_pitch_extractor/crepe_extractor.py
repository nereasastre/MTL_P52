import crepe
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def crepe_extractor(audio, sr=44100):
    """
    Extracts the fundamental frequency given an input sound using the crepe prediction method.
    Args:
        audio: the input sound (list of float)
        sr: the sampling rate (int)
    Returns:
        freq: the estimated fundamental frequency (float)
    """
    time, freqs, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    idxs = []

    for idx in range(len(freqs) - 1):
        if confidence[idx] > 0.8:
            if not idxs:
                idxs.append(idx)  # append first decent guess
        if abs(freqs[idx] - freqs[idx + 1]) > 3:
            if idx != 0:
                idxs.append(idx)

    freq = np.mean(freqs[idxs])
    return freqs


