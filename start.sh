#!/bin/bash
# AI-Enhanced PDF Extraction System - Start Script
# Startet Ollama + AI-Enhanced PDF Processor

set -e

echo "========================================================================"
echo "    AI-ENHANCED PDF EXTRACTION SYSTEM - STARTUP"
echo "========================================================================"
echo

# Check if config exists
if [[ ! -f "config.json" ]]; then
    echo "❌ config.json nicht gefunden!"
    echo "   Führen Sie zuerst die Konfiguration aus:"
    echo "   python3 setup_wizard.py"
    echo
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Ollama
check_ollama() {
    echo "🤖 OLLAMA STATUS PRÜFEN"
    echo "------------------------"
    
    if command_exists ollama; then
        echo "✅ Ollama installiert"
    else
        echo "❌ Ollama nicht installiert! Führen Sie ./install.sh aus"
        exit 1
    fi
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama Service läuft"
    else
        echo "🚀 Starte Ollama Service..."
        ollama serve &
        
        # Wait for startup
        echo "⏳ Warte auf Ollama..."
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "✅ Ollama bereit"
                break
            fi
            sleep 2
            echo -n "."
        done
        
        if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "❌ Ollama Start fehlgeschlagen"
            exit 1
        fi
    fi
    echo
}

# Check required models
check_models() {
    echo "🧠 AI MODELS PRÜFEN"
    echo "--------------------"
    
    required_models=("llama3.1:8b" "llava:7b")
    missing_models=()
    
    installed_models=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [model['name'] for model in data.get('models', [])]
    print(' '.join(models))
except:
    print('')
" 2>/dev/null || echo "")
    
    for model in "${required_models[@]}"; do
        if [[ $installed_models == *"$model"* ]]; then
            echo "✅ $model verfügbar"
        else
            echo "❌ $model fehlt"
            missing_models+=("$model")
        fi
    done
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        echo "📥 Lade fehlende Models..."
        for model in "${missing_models[@]}"; do
            echo "⏳ Lade $model..."
            ollama pull "$model"
            echo "✅ $model geladen"
        done
    fi
    echo
}

# Test AI functionality
test_ai() {
    echo "🧪 AI FUNKTIONALITÄT TESTEN"
    echo "----------------------------"
    
    # Test text model
    echo "🔤 Teste Text Model..."
    response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"llama3.1:8b","prompt":"Respond with: AI Ready","stream":false}' \
        2>/dev/null)
    
    if [[ $response == *"AI Ready"* ]]; then
        echo "✅ Text Model funktioniert"
    else
        echo "⚠️  Text Model Test unvollständig"
    fi
    
    # Test vision model
    echo "👁️  Teste Vision Model..."
    vision_response=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model":"llava:7b","prompt":"What do you see?","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="],"stream":false}' \
        2>/dev/null)
    
    if [[ $vision_response == *"response"* ]]; then
        echo "✅ Vision Model funktioniert"
    else
        echo "⚠️  Vision Model Test unvollständig"
    fi
    echo
}

# Start the PDF processor
start_processor() {
    echo "🚀 AI-ENHANCED PDF PROCESSOR STARTEN"
    echo "-------------------------------------"
    
    # Check Python dependencies
    python3 -c "
import fitz, sentence_transformers, supabase, boto3, requests
print('✅ Python Dependencies verfügbar')
" || {
        echo "❌ Python Dependencies fehlen! Führen Sie aus:"
        echo "   pip3 install -r requirements.txt"
        exit 1
    }
    
    echo "🤖 Starte AI-Enhanced PDF Extraction System..."
    echo "   📁 Überwacht: $(python3 -c "import json; print(json.load(open('config.json'))['documents_path'])" 2>/dev/null || echo "Documents/")"
    echo "   🧠 Vision AI: aktiviert"
    echo "   💭 Semantic Boundaries: aktiviert"
    echo "   🎯 Intelligent Chunking: aktiviert"
    echo
    echo "💡 Kopieren Sie PDF-Dateien in den Documents Ordner für automatische Verarbeitung"
    echo "   Drücken Sie Ctrl+C zum Beenden"
    echo "========================================================================"
    echo
    
    # Start the main processor
    python3 ai_pdf_processor.py
}

# Cleanup function
cleanup() {
    echo
    echo "🛑 System wird beendet..."
    
    # Kill background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    echo "✅ AI-Enhanced PDF Extraction System beendet"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    check_ollama
    check_models
    test_ai
    start_processor
}

# Run the startup sequence
main
