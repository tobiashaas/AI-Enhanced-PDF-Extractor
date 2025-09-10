# 🔄 Migration zu Cross-Platform Setup

## Für bestehende Nutzer

Wenn Sie bereits ein funktionierendes Setup haben, folgen Sie dieser Anleitung zur Migration auf das neue Cross-Platform System.

### ✅ Was bleibt gleich:
- Ihre verarbeiteten PDFs und Daten
- Supabase Datenbank
- R2 Storage und Images
- AI Models (llava:7b, llama3.1:8b)

### 🔧 Was sich ändert:
- Bessere Cross-Platform Kompatibilität
- Automatische Hardware-Erkennung
- Korrekte R2 Public URLs
- Vereinfachte Konfiguration

## 📋 Migrations-Schritte

### 1. Backup erstellen
```bash
cp config.json config_backup.json
cp ai_pdf_processor.py ai_pdf_processor_backup.py
```

### 2. Repository aktualisieren
```bash
git pull origin main
```

### 3. Config aktualisieren
```bash
# Option A: Automatische Migration
python3 cross_platform_setup.py

# Option B: Manuelle Update
cp config.example.json config_new.json
# Übertrage deine documents_path aus config_backup.json
# Alles andere bleibt automatisch korrekt
```

### 4. R2 URLs aktualisieren (falls nötig)
```bash
# Prüfe ob URLs korrekt sind
python3 -c "
from supabase import create_client
import json
with open('config.json') as f: config = json.load(f)
supabase = create_client(config['supabase_url'], config['supabase_key'])
result = supabase.table('images').select('r2_url').limit(1).execute()
if result.data:
    url = result.data[0]['r2_url']
    if '80a63376fddf4b909ed55ee53a401a93' in url:
        print('✅ URLs sind korrekt')
    else:
        print('⚠️ URLs müssen aktualisiert werden')
        print('Führe aus: python3 update_public_urls.py')
"
```

### 5. Hardware-Optimierung
```bash
python3 performance_optimizer.py
```

### 6. Test
```bash
python3 status.py
python3 test_presigned_url.py
```

## 🚀 Neuer PC Setup

Für einen komplett neuen PC:

```bash
git clone https://github.com/tobiashaas/AI-Enhanced-PDF-Extractor.git
cd AI-Enhanced-PDF-Extractor
pip install -r requirements.txt
python3 cross_platform_setup.py
```

Das war's! Der neue PC nutzt automatisch dieselbe Datenbank und alle bereits verarbeiteten Daten.
