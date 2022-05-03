from scipy.io import wavfile
from MTL_P52.sound_recorder.record_and_export import record_and_export
from MTL_P52.offline_pitch_extractor.crepe_extractor import crepe_pitch

"""
Records sound for a given amount of seconds and then attempts to extract pitch using crepe. 
Extracts pitch offline (after recording is over).
"""

record_and_export("")
recording_path = "../sound_recorder/output.wav"
sr, audio = wavfile.read(recording_path)

crepe_pitch(audio, sr)