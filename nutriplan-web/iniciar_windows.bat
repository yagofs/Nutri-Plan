@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0..\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Ambiente virtual do RuFaS nao encontrado em ..\venv.
    echo Instale as dependencias do projeto no ambiente existente antes de iniciar.
    pause
    exit /b 1
)
"%PYTHON%" -c "import sys, scipy; sys.path.insert(0, '..'); import RUFAS.biophysical.animal.ration.ration_optimizer" >nul 2>&1
if errorlevel 1 (
    echo Dependencias do RuFaS indisponiveis neste ambiente Python.
    echo Consulte o README do projeto para configurar o ambiente existente.
    pause
    exit /b 1
)
echo NutriPlan: http://127.0.0.1:8765
echo Pressione Ctrl+C para encerrar.
start "" http://127.0.0.1:8765
"%PYTHON%" server.py
if errorlevel 1 (
    echo Falha ao iniciar. Verifique se a porta 8765 ja esta em uso.
    pause
    exit /b 1
)
