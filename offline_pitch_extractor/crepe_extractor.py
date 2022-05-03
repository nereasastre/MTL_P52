import crepe
from scipy.io import wavfile
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# notes are E3 (130 Hz), G3 (196 Hz), F3 (174.61 Hz), C3 (130.81 Hz), C4 (261.63 Hz)


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
        if abs(frequency[idx] - frequency[idx + 1]) > 3:
            idxs.append(idx - 1)

    print("time: ", time[idxs], "frequency: ", frequency[idxs], "confidence: ", confidence[idxs])

    print(len(time), len(frequency), len(confidence), len(activation))
    return time, frequency, confidence, activation


def test_crepe_extractor():
    #todo move to test folder once we start testing
    """Tests crepe extractor against the piano.wav file"""

    sr, audio = wavfile.read("../sounds/piano.wav")
    crepe_pitch(audio, sr)


test_crepe_extractor()
