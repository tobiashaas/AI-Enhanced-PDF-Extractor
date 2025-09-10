#!/bin/bash

# 🚀 AI-Enhanced PDF Extractor - Schnell-Setup für neue Umgebungen
# Automatische Installation und Konfiguration für Linux/macOS/Windows

set -e

echo "🚀 AI-Enhanced PDF Extractor - Setup Wizard"
echo "============================================="
echo ""

# Farben für bessere Lesbarkeit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funktionen
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# System-Erkennung
detect_system() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        SYSTEM="macos"
        print_info "macOS erkannt"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        SYSTEM="linux"
        print_info "Linux erkannt"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        SYSTEM="windows"
        print_info "Windows (Git Bash/WSL) erkannt"
    else
        print_error "Unbekanntes System: $OSTYPE"
        exit 1
    fi
}

# Python-Version prüfen
check_python() {
    print_info "Prüfe Python Installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
            print_success "Python $PYTHON_VERSION gefunden"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.9+ erforderlich, gefunden: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 nicht gefunden!"
        print_info "Bitte installiere Python 3.9+ von https://python.org"
        exit 1
    fi
}

# Dependencies installieren
install_dependencies() {
    print_info "Installiere Python Dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install --upgrade pip
        $PYTHON_CMD -m pip install -r requirements.txt
        print_success "Dependencies installiert"
    else
        print_error "requirements.txt nicht gefunden!"
        exit 1
    fi
}

# Ollama Installation prüfen
check_ollama() {
    print_info "Prüfe Ollama Installation..."
    
    if command -v ollama &> /dev/null; then
        print_success "Ollama gefunden"
        
        # Prüfe ob Ollama läuft
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama Server läuft"
        else
            print_warning "Ollama Server nicht erreichbar"
            print_info "Starte Ollama..."
            
            if [[ "$SYSTEM" == "macos" ]]; then
                nohup ollama serve > /dev/null 2>&1 &
            else
                nohup ollama serve > /dev/null 2>&1 &
            fi
            
            sleep 3
            
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                print_success "Ollama Server gestartet"
            else
                print_error "Konnte Ollama Server nicht starten"
                print_info "Bitte starte 'ollama serve' manuell"
            fi
        fi
    else
        print_error "Ollama nicht gefunden!"
        print_info "Installation:"
        
        if [[ "$SYSTEM" == "macos" ]]; then
            echo "  brew install ollama"
        elif [[ "$SYSTEM" == "linux" ]]; then
            echo "  curl -fsSL https://ollama.ai/install.sh | sh"
        else
            echo "  Besuche: https://ollama.ai"
        fi
        exit 1
    fi
}

# AI Models herunterladen
download_models() {
    print_info "Prüfe AI Models..."
    
    # Prüfe verfügbare Models
    MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")
    
    # Text Model prüfen
    if echo "$MODELS" | grep -q "llama3.1:8b"; then
        print_success "Text Model (llama3.1:8b) verfügbar"
    else
        print_info "Lade Text Model herunter..."
        ollama pull llama3.1:8b
        print_success "Text Model installiert"
    fi
    
    # Vision Model prüfen
    if echo "$MODELS" | grep -q "llava:7b"; then
        print_success "Vision Model (llava:7b) verfügbar"
    else
        print_info "Lade Vision Model herunter..."
        ollama pull llava:7b
        print_success "Vision Model installiert"
    fi
}

# Hardware-Optimierung
optimize_hardware() {
    print_info "Führe Hardware-Optimierung durch..."
    
    if [ -f "performance_optimizer.py" ]; then
        $PYTHON_CMD performance_optimizer.py
        
        if [ -f "config_optimized.json" ]; then
            print_success "Hardware-optimierte Konfiguration erstellt"
        else
            print_warning "Optimierung fehlgeschlagen, verwende Standard-Config"
        fi
    else
        print_warning "performance_optimizer.py nicht gefunden"
    fi
}

# Konfiguration erstellen
setup_config() {
    print_info "Setup Konfiguration..."
    
    if [ ! -f "config.json" ]; then
        if [ -f "config_optimized.json" ]; then
            cp config_optimized.json config.json
            print_success "Optimierte Konfiguration übernommen"
        elif [ -f "config.example.json" ]; then
            cp config.example.json config.json
            print_warning "Standard-Konfiguration erstellt"
            print_info "Bitte bearbeite config.json für deine Cloud-Services"
        else
            print_error "Keine Konfigurationsvorlage gefunden!"
            exit 1
        fi
    else
        print_success "Konfiguration bereits vorhanden"
    fi
}

# Interaktiver Setup
interactive_setup() {
    echo ""
    echo "🔧 Möchtest du den interaktiven Setup-Wizard starten? [j/N]"
    read -r response
    
    if [[ "$response" =~ ^[Jj]$ ]]; then
        if [ -f "setup_wizard.py" ]; then
            print_info "Starte Setup-Wizard..."
            $PYTHON_CMD setup_wizard.py
        else
            print_error "setup_wizard.py nicht gefunden!"
        fi
    else
        print_info "Überspringe interaktiven Setup"
        print_warning "Vergiss nicht, config.json für deine Cloud-Services zu bearbeiten!"
    fi
}

# Test-Lauf
test_system() {
    echo ""
    echo "🧪 Möchtest du einen System-Test durchführen? [j/N]"
    read -r response
    
    if [[ "$response" =~ ^[Jj]$ ]]; then
        if [ -f "status.py" ]; then
            print_info "Führe System-Test durch..."
            $PYTHON_CMD status.py
        else
            print_error "status.py nicht gefunden!"
        fi
    fi
}

# Hauptprogramm
main() {
    detect_system
    check_python
    install_dependencies
    check_ollama
    download_models
    optimize_hardware
    setup_config
    interactive_setup
    test_system
    
    echo ""
    echo "🎉 Setup abgeschlossen!"
    echo ""
    print_success "Das AI-Enhanced PDF Extraction System ist bereit!"
    echo ""
    echo "Nächste Schritte:"
    echo "1. Bearbeite config.json für deine Cloud-Services (Supabase + R2)"
    echo "2. Starte: python3 ai_pdf_processor.py"
    echo "3. Lege PDFs in den Documents/ Ordner"
    echo ""
    echo "📚 Weitere Hilfe: siehe README.md"
}

# Script ausführen
main
