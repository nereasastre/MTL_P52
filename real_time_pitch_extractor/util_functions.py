import numpy as np


CONCERT_PITCH = 440
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def find_closest_note(pitch_detected):
    i = int(np.round(np.log2(pitch_detected / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    closest_pitch = round(closest_pitch, 2)
    diff = pitch_detected - closest_pitch
    diff = round(diff, 2)
    return closest_note, closest_pitch, diff
