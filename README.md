## ⚡ Quickstart for Academic Review / Advisors (Windows)

To test and execute this codebase on Windows, two separate Conda environments (`masterthesis` for the RL agent/mechanics and `chatterbox` for the voice generation server) as well as a specific **Gemma model running locally in Ollama** are required.

### 1. Prerequisites
- **Miniconda / Anaconda** installed and accessible via System PATH.
- **Ollama** installed and running locally ([Download Ollama](https://ollama.com/)).
- **Model Weights (Automatic / Manual)**: The setup script will **automatically download** the 4.41 GB quantized model (`gemma-4-e2b.gguf`) from Hugging Face on run. Alternatively, you may manually download it via [this direct download link](https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) and place it into the `ollama/` directory inside this repository.

### 2. Automated Setup (Recommended)
Open a Command Prompt or PowerShell window in the repository directory and execute the included automated setup batch script:
```cmd
setup_windows.bat
```
*This automatically creates both conda environments (`masterthesis` and `chatterbox`) from the configurations in the `environments/` directory, installs required local sublibraries in editable mode, downloads the 4.41 GB Q4_K_M GGUF weights directly from Hugging Face (if not already present), and builds the custom `gemma-4-e2b` model inside Ollama using the manifest in `ollama/Modelfile`.*

**Manual Setup Alternative:**
If you prefer running the setup commands individually in your terminal:
```bat
conda env create -f environments\masterthesis.yml --force
conda env create -f environments\chatterbox.yml --force

conda activate masterthesis
pip install -e .
pip install -e .\libraries\recurrent_maskable
conda deactivate

# After placing gemma-4-e2b.gguf inside the ollama/ directory:
ollama create gemma-4-e2b -f .\ollama\Modelfile
```

### 3. Execution & Testing
Depending on which component of the repository you would like to run:

#### 🟢 Option A: Train a new agent
Activates the core environment and trains a new agent from scratch:
Parameters can be changed in the rl_agent\alphastar_transformer_agent.py file
```bat
conda activate masterthesis
python .\rl_agent\alphastar_transformer_agent.py --pretrain
python .\rl_agent\alphastar_transformer_agent.py --finetune
```
*(Note: Parameters can be changed in the rl_agent\alphastar_transformer_agent.py file)

#### 🟢 Option B: Run Interactive Game & Voice Generation (Two Terminals Required)
To run the interactive game with TTS voice features and the local AI chatbot:
1. **Terminal 1 (Voice Generation Server):**
   ```bat
   conda activate chatterbox
   python voice_server/VoiceGeneratorServer.py
   ```
2. **Terminal 2 (Game / Chat Simulation):**
   ```bat
   conda activate masterthesis
   python game/game.py
   ```
*(Note: Ensure Ollama is running in the background for LLM chatbot interactions).*

### 4. Configuration
If you plan to use the cloud-based AI chatbot fallback (instead of the local Ollama model), you will need to provide a Gemini API key:
1. Create a `.env` file in the root directory.
2. Add your API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

### 5. Game Controls
When running the interactive game, you can use the following controls:
- **Arrow Keys (Up, Down, Left, Right):** Move your character.
- **Left Alt:** Special action / Interact.
- **Number Keys (1, 2, 3, 4):** Select options/actions.
- **V:** Press and hold (or toggle) to record your voice for the voice server.
- **Enter:** Open text chat or send a typed message.

### 6. Troubleshooting
**Running Setup Scripts:** 
It is highly recommended to run `setup_windows.bat` directly from a command line (Command Prompt or PowerShell) rather than double-clicking it in File Explorer. This ensures that if an error occurs, the terminal window remains open and you can read the exact error output before it closes.




---