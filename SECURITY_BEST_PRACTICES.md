# Umgang mit sensiblen Daten - Best Practices

## Eingerichtete Sicherheitsmaßnahmen

1. **Ausschließlich Umgebungsvariablen für sensible Daten**
   - Verwendung von `.env` Dateien für alle sensiblen Daten
   - Integration von `python-dotenv` zur einfachen Verwaltung
   - Keine sensiblen Daten in Konfigurationsdateien

2. **Strikte Trennung**
   - `config.json`: Nur nicht-sensible Konfigurationswerte und Standardeinstellungen
   - `.env`: Alle sensiblen Daten (API-Schlüssel, Zugangsdaten, URLs)
   - Konfigurationsdateien ohne Anmeldeinformationen im Repository

3. **Gitignore-Konfiguration**
   - Ausschluss von `.env` Dateien aus Git
   - Bereitstellung von `.env.example` als Template
   - Ausschluss von `config.json` mit sensiblen Daten

4. **Konfigurationsmanagement**
   - Zentrale Konfigurationsverwaltung
   - Hierarchische Struktur für bessere Organisation
   - Kein Fallback auf weniger sichere Konfigurationsmethoden

## Anleitung für neue Entwickler

1. **Erste Schritte**
   - Kopieren Sie `.env.example` nach `.env`
   - Füllen Sie alle erforderlichen Werte in der `.env` Datei aus
   - Führen Sie `pip install -r requirements.txt` aus

2. **Umgebungsvariablen**

   Die folgenden Umgebungsvariablen werden vom System verwendet:

   | Variable                | Beschreibung                                | Erforderlich |
   |-------------------------|--------------------------------------------|-------------|
   | `SUPABASE_URL`          | URL zur Supabase-Instance                   | Ja          |
   | `SUPABASE_KEY`          | Supabase API-Key                           | Ja          |
   | `R2_ACCOUNT_ID`         | Cloudflare R2 Account-ID                   | Ja          |
   | `R2_ACCESS_KEY_ID`      | Cloudflare R2 Access Key                   | Ja          |
   | `R2_SECRET_ACCESS_KEY`  | Cloudflare R2 Secret Access Key            | Ja          |
   | `R2_BUCKET_NAME`        | Cloudflare R2 Bucket Name                  | Ja          |
   | `R2_PUBLIC_URL`         | Cloudflare R2 Public URL                   | Ja          |
   | `OLLAMA_BASE_URL`       | URL für Ollama API (default: localhost)    | Nein        |
   | `EMBEDDING_MODEL`       | Name des Embedding-Modells                 | Nein        |

3. **Sicherheitshinweise**
   - Teilen Sie niemals die `.env` Datei oder ihren Inhalt
   - Verwenden Sie unterschiedliche API-Schlüssel für Entwicklung und Produktion
   - Überprüfen Sie regelmäßig die Gültigkeit und Sicherheit der verwendeten Schlüssel

## Weitere Empfehlungen

1. **Key Rotation**
   - Regelmäßiges Wechseln von API-Schlüsseln
   - Dokumentation der Rotation in einem sicheren System

2. **Zugriffsbeschränkungen**
   - Verwendung von Supabase RLS für strikte Zugriffskontrolle
   - Minimale Berechtigungen für Service-Accounts

3. **Überwachung**
   - Protokollierung aller API-Zugriffsversuche
   - Warnung bei ungewöhnlichen Zugriffsmustern