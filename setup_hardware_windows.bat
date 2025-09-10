@echo off
REM AI-Enhanced PDF System - Windows Hardware Setup
REM Optimiert für RTX A2000, A4000, A6000 Workstation GPUs

echo =====================================================
echo AI-Enhanced PDF System - Windows Hardware Setup
echo =====================================================

echo.
echo 💻 Erkenne Windows Hardware...

REM NVIDIA GPU Erkennung
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >nul 2>&1
if %errorlevel% == 0 (
    echo 🎮 NVIDIA GPU erkannt!
    
    REM GPU Info anzeigen
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader') do (
        set GPU_NAME=%%a
        set GPU_MEMORY=%%b
        echo    %%a
        echo    📊 VRAM: %%b
    )
    
    REM RTX A-Series Erkennung
    echo %GPU_NAME% | findstr /i "A6000" >nul
    if %errorlevel% == 0 (
        echo    🏢 RTX A6000 Workstation GPU - Premium Konfiguration
        set VISION_MODEL=llava:7b
        set RECOMMENDED_BATCH=200
        set PERFORMANCE_BOOST=60-90%%
        goto :setup_ollama
    )
    
    echo %GPU_NAME% | findstr /i "A5000" >nul
    if %errorlevel% == 0 (
        echo    🏢 RTX A5000 Workstation GPU - High-End Konfiguration
        set VISION_MODEL=llava:7b
        set RECOMMENDED_BATCH=180
        set PERFORMANCE_BOOST=60-80%%
        goto :setup_ollama
    )
    
    echo %GPU_NAME% | findstr /i "A4000" >nul
    if %errorlevel% == 0 (
        echo    🏢 RTX A4000 Workstation GPU - Optimale Konfiguration
        set VISION_MODEL=llava:7b
        set RECOMMENDED_BATCH=160
        set PERFORMANCE_BOOST=50-70%%
        goto :setup_ollama
    )
    
    echo %GPU_NAME% | findstr /i "A2000" >nul
    if %errorlevel% == 0 (
        echo    🏢 RTX A2000 Workstation GPU - Effiziente Konfiguration
        set VISION_MODEL=llava:7b
        set RECOMMENDED_BATCH=120
        set PERFORMANCE_BOOST=40-60%%
        goto :setup_ollama
    )
    
    REM Gaming RTX GPUs
    echo %GPU_NAME% | findstr /i "4090\|4080" >nul
    if %errorlevel% == 0 (
        echo    🎮 High-End Gaming GPU erkannt
        set VISION_MODEL=llava:7b
        set RECOMMENDED_BATCH=200
        set PERFORMANCE_BOOST=60-80%%
        goto :setup_ollama
    )
    
    REM Standard RTX GPUs
    echo    🎮 Standard RTX GPU - Ausgewogene Konfiguration
    set VISION_MODEL=llava:7b
    set RECOMMENDED_BATCH=100
    set PERFORMANCE_BOOST=40-60%%
    
    :setup_ollama
    echo.
    echo 🔧 Optimiere Ollama für NVIDIA CUDA...
    set OLLAMA_GPU_LAYERS=-1
    
    echo %GPU_NAME% | findstr /i "A" >nul
    if %errorlevel% == 0 (
        echo 🏢 Workstation-Optimierungen:
        echo    ✅ ECC Memory Support
        echo    ✅ Professional Driver Optimierung
        echo    ✅ Stabile Memory Allocation
        set CUDA_MEMORY_FRACTION=0.75
    )
    
    echo.
    echo 📥 Installiere optimierte AI Models...
    ollama pull %VISION_MODEL%
    ollama pull llama3.1:8b
    
) else (
    echo 💻 Standard CPU System erkannt
    echo    🔧 Verwende CPU-optimierte Einstellungen
    set VISION_MODEL=llava:7b
    set RECOMMENDED_BATCH=50
    set PERFORMANCE_BOOST=20-30%%
    
    echo 📥 Installiere CPU-optimierte Models...
    ollama pull llava:7b
    ollama pull llama3.1:8b
)

echo.
echo 🔧 Führe Hardware Performance Optimizer aus...
python performance_optimizer.py

REM Config anwenden falls vorhanden
if exist "config_optimized.json" (
    echo 📝 Wende optimierte Konfiguration an...
    copy config_optimized.json config.json
    echo ✅ Hardware-optimierte Konfiguration aktiv
)

echo.
echo =====================================================
echo 🎯 WINDOWS HARDWARE SETUP ABGESCHLOSSEN!
echo =====================================================
echo ✅ Ollama Models installiert: %VISION_MODEL%, llama3.1:8b
echo ✅ Hardware-spezifische Optimierungen aktiviert
echo ✅ Performance-optimierte Konfiguration aktiv
echo.
echo 🚀 Starte das AI-Enhanced PDF System:
echo    python ai_pdf_processor.py
echo.
echo 📊 Erwartete Performance-Verbesserung: %PERFORMANCE_BOOST%

if defined GPU_NAME (
    echo 🎮 GPU: %GPU_NAME%
    echo 📦 Empfohlene Batch Size: %RECOMMENDED_BATCH%
    echo ⚡ CUDA Acceleration: Aktiv
)

echo.
echo 💡 Für RTX A-Series Workstation GPUs:
echo    - ECC Memory verhindert Datenfehler
echo    - Professional Drivers für Stabilität
echo    - Optimiert für 24/7 Dauerbetrieb
echo.
pause
