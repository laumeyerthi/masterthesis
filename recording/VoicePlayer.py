import sounddevice as sd
import scipy.io.wavfile as wav
import io
import numpy as np
from recording.VoiceGenerator import VoiceGenerator

class VoicePlayer:
    def __init__(self):
        self.is_playing = False
        self.voice_gen = VoiceGenerator()
        
    def play_text(self, text):
        self.voice_gen.speak_custom(text=text)

    def play_file(self, filename="voice_command.wav", blocking=False):
        try:
            self.is_playing = True
            print(f"Playing audio from {filename}...")
            
            sample_rate, data = wav.read(filename)
            
            sd.play(data, sample_rate)
            
            if blocking:
                sd.wait()
                self.is_playing = False
                print("Playback finished.")
                
        except Exception as e:
            print(f"Error playing file: {e}")
            self.is_playing = False

    def play_bytes(self, audio_bytes, blocking=False):
        try:
            self.is_playing = True
            print("Playing AI response...")
            
            byte_io = io.BytesIO(audio_bytes)
            sample_rate, data = wav.read(byte_io)
            
            sd.play(data, sample_rate)
            
            if blocking:
                sd.wait()
                self.is_playing = False
                print("Playback finished.")
                
        except Exception as e:
            print(f"Error playing audio bytes: {e}")
            self.is_playing = False

    def stop(self):
        sd.stop()
        self.is_playing = False
        print("Playback stopped.")