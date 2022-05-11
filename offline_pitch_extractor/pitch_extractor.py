from scipy.io import wavfile
import os, sys
import crepe_extractor as ce
import fft_extractor as ffte
import zero_cros_extractor as zce

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
import sound_recorder.record_and_export as rae

"""
Records sound for a given amount of seconds and then attempts to extract pitch. 
Extracts pitch offline (after recording is over).
"""


def extract_pitch(extractor=ffte.fft_pitch):
    recording_path = "../sound_recorder/output.wav"
    rae.record_and_export(recording_path)
    sr, audio = wavfile.read(recording_path)

    freq = extractor(audio, sr)
    print("Detected frequency: ", freq)


extract_pitch(zce.zero_crossing_extractor)
