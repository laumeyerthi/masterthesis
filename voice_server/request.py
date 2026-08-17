import requests
import sounddevice as sd
import soundfile as sf
import io

def play_scotty_voice(text):
    url = "http://127.0.0.1:8000/generate"
    try:
        response = requests.get(url, params={"text": text})
        
        if response.status_code == 200:
            data, fs = sf.read(io.BytesIO(response.content))
            print(f"Playing: {text}")
            sd.play(data, fs)
            # sd.wait()
        else:
            print(f"Server Error ({response.status_code}): {response.json()}")
            
    except Exception as e:
        print(f"Client Error: {e}")

# Usage
# play_scotty_voice("Laddy, the bridge is out! You'll have to go around.")