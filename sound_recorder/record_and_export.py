import sounddevice as sd
from scipy.io.wavfile import write


def record_and_export(fs=44100):
    """
    Records audio for a given amount of time and exports recording to a wav file
    Args:
        fs the sampling rate
    """
    valid_input = False
    seconds = float(
        input("Insert number of seconds to record: ")
    )  # Duration of recording, todo account for wrong inputs

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Starting your recording...")
    sd.wait()  # Wait until recording is finished
    write("sound_recorder/output.wav", fs, myrecording)  # Save as WAV file
