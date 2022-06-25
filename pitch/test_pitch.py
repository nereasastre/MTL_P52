import unittest
from scipy.io import wavfile
import numpy as np

from offline_pitch_extractor.auto_extractor import auto_extractor
from offline_pitch_extractor.crepe_extractor import crepe_extractor
from offline_pitch_extractor.fft_extractor import fft_extractor
from offline_pitch_extractor.yin_extractor import yin_extractor
from offline_pitch_extractor.zero_cross_extractor import zero_cross_extractor
import pytest
import time


AUDIO_EXPECTED = [
    ("../MTL_P52/sounds/sine-101.wav", 101),
    ("../MTL_P52/sounds/sine-440.wav", 440),
    ("../MTL_P52/sounds/sine-490.wav", 490),
    ("../MTL_P52/sounds/sine-1000.wav", 1000),
    ("../MTL_P52/sounds/soprano-E4.wav", 329),
    ("../MTL_P52/sounds/trumpet-A4.wav", 440),
    ("../MTL_P52/sounds/violin-B3.wav", 247),
    ("../MTL_P52/sounds/vibraphone-C6.wav", 1047),
    ("../MTL_P52/sounds/sawtooth-440.wav", 440),
]

AUDIO_EXPECTED1 = [
    ("../MTL_P52/sounds/sawtooth-440.wav", 440),
]


EXTRACTORS = [crepe_extractor, fft_extractor, yin_extractor, zero_cross_extractor]


def clean_frequencies(frequency):
    non_zero_freq = []
    for freq in frequency:
        if freq > 20:
            non_zero_freq.append(freq)
    return non_zero_freq


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_crepe_extractor(audio_path, expected):
    """Tests CREPE extractor against different inputs"""
    # arrange
    sr, audio = wavfile.read(audio_path)
    # act
    frequency = crepe_extractor(audio, sr)
    # assert
    assert abs(np.mean(frequency) - expected) <= 4


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_fft_extractor(audio_path, expected):
    """Tests fft extractor against different inputs"""
    # arrange
    sr, audio = wavfile.read(audio_path)
    # act
    frequency = fft_extractor(audio, sr)
    # assert
    non_zero_freq = clean_frequencies(frequency)

    assert abs(np.mean(non_zero_freq) - expected) <= 4


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_zero_cross_extractor(audio_path, expected):
    """Tests zero crossing extractor against different inputs"""

    # arrange
    sr, audio = wavfile.read(audio_path)
    # act
    frequency = zero_cross_extractor(audio, sr)
    # assert
    assert abs(frequency - expected) <= 3


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_yin_extractor(audio_path, expected):
    """Tests YIN extractor against different inputs"""

    # arrange
    sr, audio = wavfile.read(audio_path)
    # act
    frequency = yin_extractor(audio, sr)
    # assert
    non_zero_freq = clean_frequencies(frequency)
    assert abs(np.mean(non_zero_freq) - expected) <= 4


@pytest.mark.parametrize('audio_path, expected', AUDIO_EXPECTED)
def test_auto_extractor(audio_path, expected):
    """Tests autocorrelation extractor against different inputs"""
    # arrange
    sr, audio = wavfile.read(audio_path)
    # act
    frequency = auto_extractor(audio, sr)
    # assert
    assert abs(frequency - expected) <= 3


@pytest.mark.parametrize('extractor', EXTRACTORS)
def test_execution_time(extractor):
    # arrange
    audio_path = "../MTL_P52/sounds/sine-101.wav"
    sr, audio = wavfile.read(audio_path)
    start_time = time.time()

    # act
    frequency = extractor(audio, sr)
    end_time = time.time()

    # assert
    execution_time = end_time - start_time
    audio_duration = len(audio) / sr
    real_time_factor = execution_time / audio_duration
    assert real_time_factor <= 1

