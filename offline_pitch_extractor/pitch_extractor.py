from scipy.io import wavfile
import os, sys
from auto_extractor import auto_extractor
from crepe_extractor import crepe_extractor
from fft_extractor import fft_extractor
from zero_cross_extractor import zero_cross_extractor
from yin_extractor import yin_extractor
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
from sound_recorder.record_and_export import record_and_export


def extract_pitch():
    """
    Records sound for a given amount of seconds and then attempts to extract pitch.
    Extracts pitch offline (after recording is over).
    """
    extractors = [
        zero_cross_extractor,
        crepe_extractor,
        fft_extractor,
        auto_extractor,
        yin_extractor,
    ]
    extractor_names = ["Zero Crossing", "CREPE", "FFT", "Autocorrelation", "YIN"]

    # Ask for the extractor type
    extractor_idx = int(
        input(
            "Insert extractor to use: \n"
            "    1. Zero Crossing \n"
            "    2. CREPE \n"
            "    3. FFT \n"
            "    4. Autocorrelation \n"
            "    5. YIN \n"
        )
    )  # Extractor type todo account for wrong inputs

    """# record and load audio
    recording_path = "../sound_recorder/output.wav"
    record_and_export(recording_path)
    sr, audio = wavfile.read(recording_path)"""

    # record and load audio
    path = "../sounds/sine-101.wav"
    sr, audio = wavfile.read(path)

    # Extract pitch calling the corresponding extractor

    start_time = time.time()  # Measures execution time

    freq = extractors[extractor_idx - 1](audio, sr)

    # measures of efficiency
    end_time = time.time()
    execution_time = end_time - start_time

    audio_dur = len(audio) / sr
    rtf = execution_time / audio_dur

    # Print some info
    print("Execution time %s seconds" % execution_time)
    print("Real-Time factor = %f" % rtf)
    print("Detected frequency: ", freq)


extract_pitch()
