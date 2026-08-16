import torch
from faster_whisper import WhisperModel

class LocalSTT:
    def __init__(self, model_size="small"):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"  
            print("GPU detected. Faster-Whisper will run on CUDA.")
        else:
            self.device = "cpu"
            self.compute_type = "int8"     
            print("No GPU detected. Faster-Whisper will run on CPU.")

        print(f"Loading Whisper '{model_size}' model...")
        self.model = WhisperModel(
            model_size_or_path=model_size, 
            device=self.device, 
            compute_type=self.compute_type
        )
        print("STT Model loaded and ready!")

    def transcribe(self, audio_file):
        if not audio_file:
            return ""

        print("Transcribing audio...")
        segments, info = self.model.transcribe(audio_file, beam_size=3)
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
            
        final_text = full_text.strip()
        print(f"Transcription complete: '{final_text}'")
        
        return final_text