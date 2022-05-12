import unittest
from scipy.io import wavfile
import numpy as np
from offline_pitch_extractor.crepe_extractor import crepe_pitch
from offline_pitch_extractor.zero_cros_extractor import zero_crossing_extractor
import pytest


AUDIO_EXPECTED = [
    ("../MTL_P52/sounds/sine-101.wav", 101),
    ("../MTL_P52/sounds/sine-440.wav", 440),
    ("../MTL_P52/sounds/sine-490.wav", 490),
    ("../MTL_P52/sounds/sine-1000.wav", 1000),
    ("../MTL_P52/sounds/trumpet-A4.wav", 440),
    ("../MTL_P52/sounds/trumpet-A4.wav", 440),
    ("../MTL_P52/sounds/violin-B3.wav", 247),
    ]


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_crepe_extractor(audio_path, expected):
    """ Tests CREPE extractor against different inputs"""
    # arrange
    fs, audio = wavfile.read(audio_path)
    # act
    _, frequency, _, _ = crepe_pitch(audio, fs)
    # assert
    assert abs(np.mean(frequency) - expected) < 2


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_fft_extractor(audio_path, expected):
    # arrange
    fs, audio = wavfile.read(audio_path)
    # act
    # todo add frequency extractor once fft is pushed
    pass


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_zero_cros_extractor(audio_path, expected):
    # arrange
    fs, audio = wavfile.read(audio_path)
    # act
    frequency = zero_crossing_extractor(audio, fs)
    # assert
    assert abs(frequency - expected) < 2


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_yin_extractor(audio_path, expected):
    # arrange
    fs, audio = wavfile.read(audio_path)
    # act
    # todo add frequency extractor once yin is pushed
    pass


if __name__ == '__main__':
    unittest.main()
# Create your tests here.
