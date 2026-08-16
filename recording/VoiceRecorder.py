import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import queue

class VoiceRecorder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording_stream = None
        self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start(self):
        while not self.audio_queue.empty():
            self.audio_queue.get()
            
        self.is_recording = True
        print("Microphone listening...")
        
        self.recording_stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=self._audio_callback
        )
        self.recording_stream.start()

    def stop_and_save(self, filename="voice_command.wav"):
        self.is_recording = False
        print("Microphone stopped. Processing...")
        
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()
            
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
            
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0)
            wav.write(filename, self.sample_rate, audio_data)
            print(f"Saved audio to {filename}")
            return filename
        
        print("Error: No audio captured.")
        return None