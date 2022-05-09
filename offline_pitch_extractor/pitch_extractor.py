from scipy.io import wavfile

import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

from offline_pitch_extractor import crepe_extractor as ce
from sound_recorder import record_and_export as rae

"""
Records sound for a given amount of seconds and then attempts to extract pitch using crepe. 
Extracts pitch offline (after recording is over).
"""


def extract_pitch(extractor=ce.crepe_pitch):

    rae.record_and_export()
    recording_path = "../sound_recorder/output.wav"
    sr, audio = wavfile.read(recording_path)

    extractor(audio, sr)
