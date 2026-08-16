@echo off
echo ========================================================
echo        Riddle-Generator Advisor Setup (Windows)
echo ========================================================
echo.
echo [1/4] Creating/Updating conda environment: masterthesis...
call conda env create -f environments\masterthesis.yml --force
if %errorlevel% neq 0 (
    echo [ERROR] Failed to set up 'masterthesis' conda environment. Ensure Conda is installed and added to PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/4] Creating/Updating conda environment: chatterbox...
call conda env create -f environments\chatterbox.yml --force
if %errorlevel% neq 0 (
    echo [ERROR] Failed to set up 'chatterbox' conda environment.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/4] Installing root project and sublibraries into 'masterthesis'...
call conda activate masterthesis
call pip install -e .
call pip install -e .\libraries\recurrent_maskable
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install editable local packages into 'masterthesis'.
    pause
    exit /b %errorlevel%
)
call conda deactivate

echo.
echo [4/4] Setting up local custom Gemma model (gemma-4-e2b) in Ollama...
if not exist "ollama\gemma-4-e2b.gguf" (
    echo [INFO] Custom model file (gemma-4-e2b.gguf) not found in 'ollama\' directory.
    echo Downloading 4.41 GB Q4_K_M model directly from Hugging Face...
    echo (This may take several minutes depending on your connection speed...)
    powershell -Command "if (!(Test-Path 'ollama')) { New-Item -ItemType Directory -Force -Path 'ollama' }; Invoke-WebRequest -Uri 'https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf' -OutFile 'ollama\gemma-4-e2b.gguf'"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to download model automatically. Please download manually from:
        echo https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf
        echo and place it into the 'ollama\' directory as 'gemma-4-e2b.gguf'.
        pause
        exit /b %errorlevel%
    )
)

echo Registering custom model with Ollama...
call ollama create gemma-4-e2b -f .\ollama\Modelfile
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create model in Ollama. Ensure Ollama is installed and currently running.
) else (
    echo [SUCCESS] gemma-4-e2b registered successfully in Ollama!
)

echo.
echo ========================================================
echo Setup routine finished! You are ready to run the project.
echo ========================================================
pause
