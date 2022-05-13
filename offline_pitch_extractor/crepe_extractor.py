import crepe
import os
import statistics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def crepe_pitch(audio, sr):
    """
    Extracts pitch from sound and returns
    Args:
        audio: a .wav file containing the audio from which to extract pitch
        sr: the sampling rate of the audio
    Output:
        Returns time (the time onsets), frequency (the pitch),
        confidence (the prediction confidence), activation (the activation)

    """
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    idxs = []

    for idx in range(len(frequency) - 1):
        if confidence[idx] > 0.8:
            if not idxs:
                idxs.append(idx)  # append first decent guess
        if abs(frequency[idx] - frequency[idx + 1]) > 3:
            if idx != 0:
                idxs.append(idx)

    # print("time: ", time[idxs], "frequency: ", frequency[idxs], "confidence: ", confidence[idxs])
    # print("time: ", time, "frequency: ", frequency, "confidence: ", confidence)
    # print(len(audio), "\n", len(time), "\n",  len(frequency), "\n",len(confidence),  "\n", len(activation))
    # return time, frequency, confidence, activation
    freq = statistics.mean(frequency[idxs])
    return freq
