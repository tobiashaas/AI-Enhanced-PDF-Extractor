# 🔧 TECHNICAL REQUIREMENTS
**AI-Enhanced PDF Processing System - Refactor Branch**
*Letzte Aktualisierung: 13. September 2025*

## 📦 **OLLAMA SETUP**

### 🎯 **Version Requirements:**
```bash
# Minimum Version für embeddinggemma support
Ollama >= 0.11.10

# Prüfen der aktuellen Version
ollama --version
```

### 🤖 **Embedding Model:**
```bash
# Model downloaden
ollama pull embeddinggemma

# Model info
ollama show embeddinggemma
```

### 🔌 **API Configuration:**

#### **Neue Embedding API (verwenden!):**
```bash
# Endpoint
POST http://localhost:11434/api/embed

# Request Format
{
  "model": "embeddinggemma",
  "input": "Text to embed for semantic search..."
}

# Response Format
{
  "embeddings": [[0.1, 0.2, ...]] // 768-dimensional array
}
```

#### **⚠️ Legacy API (NICHT verwenden!):**
```bash
# DEPRECATED - nicht mehr nutzen
POST http://localhost:11434/api/embeddings
```

### 🚀 **Performance Optimization:**

```bash
# Batch Processing für bessere Performance
{
  "model": "embeddinggemma",
  "input": [
    "First text chunk...",
    "Second text chunk...",
    "Third text chunk..."
  ]
}

# Response mit mehreren Embeddings
{
  "embeddings": [
    [0.1, 0.2, ...], // First chunk embedding
    [0.3, 0.4, ...], // Second chunk embedding
    [0.5, 0.6, ...]  // Third chunk embedding
  ]
}
```

---

## 🗄️ **DATABASE VECTOR SETUP**

### 📊 **Vector Dimensions:**
```sql
-- Alle embedding Spalten sind 768-dimensional
embedding vector(768)

-- Für EmbeddingGemma optimiert
-- Google's modernste Embedding-Technologie (September 2025)
```

### 🔍 **Vector Search Queries:**
```sql
-- Semantic Search mit Cosine Similarity
SELECT content, manufacturer, model
FROM service_manuals 
WHERE manufacturer = 'HP' 
ORDER BY embedding <-> $1::vector 
LIMIT 10;

-- Mit Similarity Threshold
SELECT content, 
       (1 - (embedding <-> $1::vector)) as similarity
FROM service_manuals 
WHERE (1 - (embedding <-> $1::vector)) > 0.7
ORDER BY similarity DESC;
```

---

## 🐳 **DOCKER SETUP**

### 🔧 **Ollama Container:**
```dockerfile
# Dockerfile für Ollama Service
FROM ollama/ollama:latest

# Health Check für API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:11434/api/tags || exit 1

EXPOSE 11434
```

### 📝 **Docker Compose Integration:**
```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  ollama_data:
```

---

## ⚙️ **ENVIRONMENT CONFIGURATION**

### 🔑 **Environment Variables:**
```bash
# Environment Variables
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=embeddinggemma

# API Endpoints
OLLAMA_EMBED_ENDPOINT=http://localhost:11434/api/embed
OLLAMA_CHAT_ENDPOINT=http://localhost:11434/api/chat

# Performance Settings
OLLAMA_BATCH_SIZE=10
OLLAMA_TIMEOUT=30000
OLLAMA_MAX_RETRIES=3
```

### 📋 **Config.json Example:**
```json
{
  "ollama": {
    "host": "localhost",
    "port": 11434,
    "model": "embeddinggemma",
    "api": {
      "embed": "/api/embed",
      "chat": "/api/chat"
    },
    "settings": {
      "batch_size": 10,
      "timeout": 30000,
      "max_retries": 3,
      "vector_dimensions": 768
    }
  },
  "database": {
    "vector_similarity_threshold": 0.7,
    "max_search_results": 50
  }
}
```

---

## 🧪 **TESTING**

### ✅ **Ollama Health Check:**
```bash
# API verfügbar?
curl http://localhost:11434/api/tags

# Model geladen?
curl http://localhost:11434/api/show -d '{"name": "embeddinggemma"}'

# Embedding Test
curl http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embeddinggemma",
    "input": "Test embedding generation"
  }'
```

### 🔍 **Vector Search Test:**
```sql
-- Test Embedding Insert
INSERT INTO service_manuals (content, manufacturer, model, embedding)
VALUES (
  'Test content for embedding',
  'HP',
  'E55040',
  '[0.1, 0.2, 0.3, ...]'::vector(768)
);

-- Test Vector Search
SELECT content FROM service_manuals 
ORDER BY embedding <-> '[0.1, 0.2, 0.3, ...]'::vector(768) 
LIMIT 5;
```

---

## 🚨 **TROUBLESHOOTING**

### ❌ **Common Issues:**

1. **Ollama Version zu alt:**
   ```bash
   # Update Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama --version
   ```

2. **Model nicht gefunden:**
   ```bash
   # Model neu downloaden
   ollama pull embeddinggemma
   ollama list
   ```

3. **API Endpoint nicht erreichbar:**
   ```bash
   # Ollama Service prüfen
   ps aux | grep ollama
   curl http://localhost:11434/api/tags
   ```

4. **Vector Dimension Mismatch:**
   ```sql
   -- Prüfen der Vector Dimensionen
   SELECT vector_dims(embedding) FROM service_manuals LIMIT 1;
   -- Sollte 768 zurückgeben
   ```

---

## 📚 **DOCUMENTATION LINKS**

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [EmbeddingGemma Model](https://ollama.ai/library/embeddinggemma)
- [Supabase Vector Documentation](https://supabase.com/docs/guides/ai/vector-embeddings)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)

---

**✅ Mit dieser Konfiguration ist das System optimal für AI-Enhanced PDF Processing setup!**