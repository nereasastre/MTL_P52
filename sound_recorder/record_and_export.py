import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
valid_input = False
seconds = float(input("Insert number of seconds to record: ")) # Duration of recording, todo account for wrong inputs

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print("Starting your recording...")
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file