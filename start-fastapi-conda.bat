@echo off
setlocal

set "AI_PORT=%~1"
set "CONDA_ENV=%~2"

if "%AI_PORT%"=="" set "AI_PORT=8002"
if "%CONDA_ENV%"=="" set "CONDA_ENV=telucup-ai"

call conda activate "%CONDA_ENV%"

python -c "import sys; print('[INFO] Python executable:', sys.executable); raise SystemExit(0 if '%CONDA_ENV%' in sys.executable else 1)"
if errorlevel 1 (
    echo [WARN] Conda env %CONDA_ENV% tidak memakai python env yang benar.
    echo [WARN] Saat ini Python kemungkinan masih dari base Anaconda.
    echo [WARN] Chatbot tetap dicoba dijalankan, tetapi face recognition bisa gagal jika Torch/TorchVision mismatch.
    echo [WARN] Perbaikan yang disarankan: recreate env dengan Python 3.12 lalu install requirements.txt.
)

python -m uvicorn main:app --host 127.0.0.1 --port %AI_PORT% --reload
