from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
import sys
import logging
logging.basicConfig(level=logging.INFO)

import sys
import logging
import io
import os
import uuid
import soundfile as sf
import torch
import torchaudio
import perth
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from pathlib import Path



def _soundfile_load(filepath, **kwargs):
    audio, sr = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(audio).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

torchaudio.load = _soundfile_load

if perth.PerthImplicitWatermarker is None:
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

from chatterbox.tts_turbo import ChatterboxTurboTTS

class VoiceGenerator:
    def __init__(self):
        print("--- [INFO] Loading Chatterbox Model ---")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- [INFO] Active Device: {self.device} ---")
        self.tts = ChatterboxTurboTTS.from_pretrained(self.device) 
        root_dir = Path(__file__).parent.parent
        self.ref_audio = root_dir / "voicelines" / "scotty_scott_uncleaned.wav"
        
        print("--- [INFO] Preparing Voice Conditionals ---")
        self.tts.prepare_conditionals(self.ref_audio)

    def generate_bytes(self, text):
        with torch.no_grad():
            audio_tensor = self.tts.generate(text=text)
        
        byte_io = io.BytesIO()
        audio_data = audio_tensor.cpu().float()
        if audio_data.ndim == 1:
            audio_data = audio_data.unsqueeze(0)
            
        sf.write(byte_io, audio_data.numpy().T, 24000, format='WAV')
        byte_io.seek(0)
        return byte_io

app = FastAPI()
generator = VoiceGenerator()

@app.get("/generate")
async def generate(text: str):
    try:
        audio_stream = generator.generate_bytes(text)
        return StreamingResponse(audio_stream, media_type="audio/wav")
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)