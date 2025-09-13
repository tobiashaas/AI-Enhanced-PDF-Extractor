# Windows-Installationsanleitung für PDF-Extractor

Diese Anleitung beschreibt die Installation und Konfiguration des PDF-Extractors unter Windows.

## Voraussetzungen

Bevor Sie beginnen, stellen Sie sicher, dass die folgenden Komponenten auf Ihrem System installiert sind:

1. **Python 3.9 oder neuer**: [Python herunterladen](https://www.python.org/downloads/)
   - Bei der Installation die Option "Add Python to PATH" aktivieren
   
2. **Git**: [Git für Windows herunterladen](https://gitforwindows.org/)

3. **Visual C++ Build Tools**: Für einige Pakete wie PyMuPDF und psycopg2-binary
   - [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Bei der Installation "Desktop development with C++" auswählen

4. **Ollama** (Zwei Optionen):
   - **Option A**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) und dann Ollama als Container ausführen
   - **Option B**: [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) und dann Ollama unter Linux installieren

## Installation

1. **Repository klonen**

   ```cmd
   git clone https://github.com/IhrRepository/PDF-Extractor.git
   cd PDF-Extractor
   ```

2. **Python-Umgebung einrichten**

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Abhängigkeiten installieren**

   ```cmd
   pip install -r requirements.txt
   ```

   Falls Fehler mit `psycopg2-binary` auftreten:
   ```cmd
   pip install psycopg2-binary --no-binary :all:
   ```

4. **Umgebungsvariablen einrichten**

   ```cmd
   copy .env.example .env
   ```

   Die `.env`-Datei in einem Texteditor öffnen und alle erforderlichen Werte eintragen. Achten Sie besonders auf die korrekten Windows-Pfadangaben mit doppelten Backslashes:

   ```
   DOCUMENTS_PATH=C:\\Users\\username\\PDF-Extractor\\Documents
   MEMORY_DIR=C:\\Users\\username\\PDF-Extractor\\MEMORY
   ```

## Ollama konfigurieren

### Option A: Mit Docker

1. Docker Desktop installieren und starten
2. Ollama-Container starten:

   ```cmd
   docker run -d --name ollama -p 11434:11434 ollama/ollama
   ```

3. EmbeddingGemma-Modell herunterladen:

   ```cmd
   curl -X POST http://localhost:11434/api/pull -d '{"name": "embeddinggemma"}'
   ```

### Option B: Mit WSL2

1. WSL2 installieren und eine Linux-Distribution einrichten
2. Ollama innerhalb von WSL installieren:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

3. Ollama starten und das EmbeddingGemma-Modell installieren:

   ```bash
   ollama serve &
   ollama pull embeddinggemma
   ```

4. In der `.env`-Datei den OLLAMA_BASE_URL-Eintrag anpassen:

   ```
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## Häufige Probleme und Lösungen

### "Images"-Tabellen-Probleme

Wenn Fehler bezüglich der "images"-Tabelle auftreten:

```cmd
python fix_images_table.py
```

Oder führen Sie das SQL-Skript manuell in der Supabase-Konsole aus:

1. Öffnen Sie die Supabase SQL-Editor
2. Kopieren Sie den Inhalt der Datei 'fix_images_table.sql'
3. Führen Sie den SQL-Code aus

### PyMuPDF-Installationsprobleme

Wenn beim Installieren von PyMuPDF Fehler auftreten:

```cmd
pip uninstall PyMuPDF
pip install --upgrade pip setuptools wheel
pip install PyMuPDF==1.23.26
```

### R2/Cloudflare-Verbindungsprobleme

Wenn Probleme mit der R2-Verbindung auftreten:

1. Prüfen Sie die Firewall-Einstellungen
2. Stellen Sie sicher, dass alle Umgebungsvariablen für R2 korrekt gesetzt sind
3. Überprüfen Sie den Netzwerkzugriff auf die Cloudflare-Endpunkte

### Supabase-Verbindungsprobleme

Bei Supabase-Verbindungsproblemen:

1. Prüfen Sie die Supabase-URL und den API-Schlüssel
2. Führen Sie einen einfachen Verbindungstest aus:

   ```cmd
   python db_check.py
   ```

## Anwendung ausführen

Nach erfolgreicher Installation können Sie das PDF-Extractor-System starten:

```cmd
python ai_pdf_processor.py
```

## Support

Falls weiterhin Probleme auftreten, erstellen Sie bitte ein Issue im GitHub-Repository mit folgenden Informationen:

- Genaue Fehlermeldung
- Windows-Version
- Python-Version
- Installierte Pakete (`pip freeze > packages.txt` und die Datei anhängen)