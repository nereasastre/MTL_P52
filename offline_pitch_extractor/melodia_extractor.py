from scipy.io import wavfile
from essentia.standard import *
import IPython


# Plots

import matplotlib

matplotlib.use("Agg")
from pylab import plot, show, figure, imshow

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 6)
try:
    # for Python2
    from Tkinter import *  ## notice capitalized T in Tkinter
except ImportError:
    # for Python3
    from tkinter import *  ## notice lowercase 't' in tkinter here

import numpy

w = Windowing(type="hann")
spectrum = (
    Spectrum()
)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()
audiofile = "../sounds/sine-101.wav"

loader = EqloudLoader(filename=audiofile, sampleRate=44100)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio) / 44100.0)

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).

pitch_extractor = PredominantPitchMelodia(frameSize=2048, hopSize=128)
pitch_values, pitch_confidence = pitch_extractor(audio)

# Pitch is estimated on frames. Compute frame time positions.
pitch_times = numpy.linspace(0.0, len(audio) / 44100.0, len(pitch_values))
print(pitch_times, pitch_values)
# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title("estimated pitch [Hz]")
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title("pitch confidence")
plt.show()
