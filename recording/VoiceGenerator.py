
import torchaudio
import torch
import soundfile as sf
import perth
import os

if perth.PerthImplicitWatermarker is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

def _soundfile_load(filepath, **kwargs):
    audio, sr = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(audio)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

def _soundfile_save(filepath, tensor, sample_rate, **kwargs):
    if tensor.ndim == 2:
        tensor = tensor.transpose(0, 1)
    sf.write(filepath, tensor.numpy(), sample_rate)

torchaudio.load = _soundfile_load
torchaudio.save = _soundfile_save

from chatterbox.tts_turbo import ChatterboxTurboTTS
import sounddevice as sd

class VoiceGenerator:
    
    def __init__(self):
        print("Loading Chatterbox Turbo TTS model... (This may take a moment on first run)")
        self.tts = ChatterboxTurboTTS.from_pretrained('cuda' if torch.cuda.is_available() else 'cpu') 
        
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.ref_audio = os.path.join(root_dir, "voicelines", "scooty_scott_uncleaned.wav")
        
        self.ref_text = "Laddy, I was drinking scotch a hundred years before you were born, and I can tell you that whatever this is, it is definitely not scotch."

    def speak_custom(self, text):
        audio_tensor = self.tts.generate(
            text=text,
            audio_prompt_path=self.ref_audio
        )
        sample_rate = 24000 
        
        if isinstance(audio_tensor, torch.Tensor):
            audio_data = audio_tensor.squeeze().cpu().numpy()
        else:
            audio_data = audio_tensor
        
        sd.play(audio_data, sample_rate)
        
        self.save_audio(audio_tensor, sample_rate, "voice_response.wav")

    def save_audio(self, audio_data, sample_rate, filename):
        try:
            if isinstance(audio_data, torch.Tensor):
                audio_to_save = audio_data.cpu()
            else:
                audio_to_save = torch.from_numpy(audio_data)
            
            if audio_to_save.ndim == 1:
                audio_to_save = audio_to_save.unsqueeze(0)
            
            torchaudio.save(filename, audio_to_save, sample_rate)
            print(f"[SUCCESS] Scotty's voice saved to: {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save audio: {e}")