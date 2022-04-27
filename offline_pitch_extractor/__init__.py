import crepe
from scipy.io import wavfile

sr, audio = wavfile.read("C:/Users/nersa/OneDrive/Documents/MTL/MTL_P52/sounds/piano.wav")
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
