#!/bin/bash
# AI-Enhanced PDF Extraction System - Installation Script
# Automatische Installation von Ollama + Models + Python Dependencies

set -e

echo "========================================================================"
echo "    AI-ENHANCED PDF EXTRACTION SYSTEM - INSTALLATION"
echo "========================================================================"
echo "Installiert Ollama, lädt AI Models herunter und richtet das System ein"
echo "========================================================================"
echo

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
else
    echo "❌ Nicht unterstütztes Betriebssystem: $OSTYPE"
    exit 1
fi

echo "🖥️  Platform erkannt: $PLATFORM"
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Ollama if not present
install_ollama() {
    echo "🤖 OLLAMA INSTALLATION"
    echo "----------------------"
    
    if command_exists ollama; then
        echo "✅ Ollama ist bereits installiert"
        ollama --version
    else
        echo "📥 Installiere Ollama..."
        if [[ "$PLATFORM" == "macOS" ]]; then
            # macOS installation
            if command_exists brew; then
                brew install ollama
            else
                curl -fsSL https://ollama.ai/install.sh | sh
            fi
        else
            # Linux installation
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
        
        echo "✅ Ollama erfolgreich installiert"
    fi
    echo
}

# Start Ollama service
start_ollama() {
    echo "🚀 OLLAMA SERVICE STARTEN"
    echo "-------------------------"
    
    # Check if Ollama is already running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama läuft bereits"
    else
        echo "🔄 Starte Ollama Service..."
        if [[ "$PLATFORM" == "macOS" ]]; then
            ollama serve &
        else
            systemctl start ollama || ollama serve &
        fi
        
        # Wait for Ollama to start
        echo "⏳ Warte auf Ollama Startup..."
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "✅ Ollama ist bereit"
                break
            fi
            sleep 2
            echo -n "."
        done
        
        if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "❌ Ollama konnte nicht gestartet werden"
            exit 1
        fi
    fi
    echo
}

# Download required AI models
download_models() {
    echo "📥 AI MODELS HERUNTERLADEN"
    echo "--------------------------"
    
    # Check which models are already installed
    installed_models=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [model['name'] for model in data.get('models', [])]
    print(' '.join(models))
except:
    print('')
")
    
    echo "Installierte Models: $installed_models"
    
    # Required models
    declare -a models=("llama3.1:8b" "llava:7b")
    
    for model in "${models[@]}"; do
        if [[ $installed_models == *"$model"* ]]; then
            echo "✅ $model bereits vorhanden"
        else
            echo "📥 Lade $model herunter (kann einige Minuten dauern)..."
            ollama pull "$model"
            echo "✅ $model erfolgreich heruntergeladen"
        fi
    done
    echo
}

# Install Python dependencies
install_python_deps() {
    echo "🐍 PYTHON DEPENDENCIES INSTALLIEREN"
    echo "------------------------------------"
    
    if command_exists python3; then
        echo "✅ Python3 gefunden: $(python3 --version)"
    else
        echo "❌ Python3 nicht gefunden! Bitte installieren Sie Python 3.9+"
        exit 1
    fi
    
    if command_exists pip3; then
        echo "✅ pip3 gefunden"
    else
        echo "❌ pip3 nicht gefunden! Bitte installieren Sie pip"
        exit 1
    fi
    
    echo "📦 Installiere Python Packages..."
    pip3 install -r requirements.txt
    echo "✅ Python Dependencies installiert"
    echo
}

# Create directories
setup_directories() {
    echo "📁 VERZEICHNISSE EINRICHTEN"
    echo "----------------------------"
    
    mkdir -p Documents
    mkdir -p logs
    
    echo "✅ Documents/ Verzeichnis erstellt"
    echo "✅ logs/ Verzeichnis erstellt"
    echo
}

# Test installation
test_installation() {
    echo "🧪 INSTALLATION TESTEN"
    echo "-----------------------"
    
    # Test Ollama
    echo "🤖 Teste Ollama API..."
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama API erreichbar"
    else
        echo "❌ Ollama API nicht erreichbar"
        return 1
    fi
    
    # Test Models
    echo "🧠 Teste AI Models..."
    response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"llama3.1:8b","prompt":"Test","stream":false}' \
        2>/dev/null)
    
    if [[ $response == *"response"* ]]; then
        echo "✅ Text Model (llama3.1:8b) funktioniert"
    else
        echo "⚠️  Text Model Test fehlgeschlagen"
    fi
    
    # Test Python imports
    echo "🐍 Teste Python Dependencies..."
    python3 -c "
try:
    import fitz
    import sentence_transformers
    import supabase
    import boto3
    import requests
    print('✅ Alle Python Dependencies verfügbar')
except ImportError as e:
    print(f'❌ Import Fehler: {e}')
    exit(1)
" || return 1
    
    echo "✅ Installation erfolgreich getestet"
    echo
}

# Main installation
main() {
    install_ollama
    start_ollama
    download_models
    install_python_deps
    setup_directories
    test_installation
    
    echo "========================================================================"
    echo "    INSTALLATION ERFOLGREICH ABGESCHLOSSEN!"
    echo "========================================================================"
    echo "✅ Ollama Service läuft"
    echo "✅ AI Models heruntergeladen (llama3.1:8b, llava:7b)"
    echo "✅ Python Dependencies installiert"
    echo "✅ Verzeichnisse eingerichtet"
    echo
    echo "🚀 NÄCHSTE SCHRITTE:"
    echo "1. Konfiguration: python3 setup_wizard.py"
    echo "2. System starten: python3 ai_pdf_processor.py"
    echo "3. PDFs in Documents/ Ordner kopieren"
    echo
    echo "📊 SYSTEM BEREIT FÜR AI-ENHANCED PDF PROCESSING!"
    echo "========================================================================"
}

# Run installation
main
