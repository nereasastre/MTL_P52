import crepe
from scipy.io import wavfile
import os
import librosa
import aubio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# notes are E3 (130 Hz), G3 (196 Hz), F3 (174.61 Hz), C3 (130.81 Hz), C4 (261.63 Hz)
sr, audio = wavfile.read("../sounds/piano.wav")
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
crepe.predict
idxs = []
for idx in range(len(frequency) - 1):
    if abs(frequency[idx] - frequency[idx + 1]) > 3:
        idxs.append(idx - 1)

print("time: ", time[idxs], "frequency: ", frequency[idxs], "confidence: ", confidence[idxs])

"""
print(time, frequency, confidence, activation)
"""
print(len(time), len(frequency), len(confidence), len(activation))

