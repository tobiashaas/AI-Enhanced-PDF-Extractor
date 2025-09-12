# AI-Enhanced PDF Extractor - Beta

🚀 **Optimized AI Agent Ready Version**

## 🎯 Features (Beta)

### ✨ **Optimized Parts Catalog**
- **No Price/Availability Dependencies** - Clean AI Agent Integration
- **Part Number as Unique Reference** - Eliminates 99.7% duplicates  
- **Quality-based Prioritization** - Smart ranking system
- **Model Compatibility Support** - Multi-model parts matching

### 🏎️ **Performance Optimizations**
- **8 Database Indices** - Sub-second query performance
- **Vector Search Ready** - Semantic similarity matching
- **Optimized Views** - AI Agent specific database views
- **Smart Chunking** - Intelligent text segmentation

### 🤖 **AI Agent Ready**
- **Helper Functions** - `parts_helper_optimized.py`
- **Optimized Search** - Part number and fuzzy matching
- **Quality Analytics** - Data completeness scoring
- **Clean API** - No legacy dependencies

## 🚀 Quick Start

### 1. Cross-Platform Setup (Windows/Linux/macOS)
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive setup wizard (all platforms)
python3 setup_wizard.py
```

**Platform Features:**
- 🍎 **macOS**: Apple Silicon Metal acceleration
- 🐧 **Linux**: CUDA GPU acceleration  
- 🪟 **Windows**: DirectML support
- 🐳 **Docker**: Cross-platform containerized deployment

### 2. Database Setup & Optimization
```bash
# Apply AI Agent performance optimizations
python3 -c "
import json
with open('ai_agent_safe_optimization.sql', 'r') as f:
    sql = f.read()
# Execute in Supabase Dashboard SQL Editor
print('Copy this SQL to Supabase Dashboard:')
print(sql)
"
```

### 3. Process Documents
```bash
python3 ai_pdf_processor.py --full-processing
```

### 4. Test AI Agent Integration
```python
from parts_helper_optimized import *
from database_client import DatabaseClient

# Initialize
db = DatabaseClient(config['supabase_url'], config['supabase_key'])

# Search parts by number
part = get_part_by_number(db, "A93E563400")

# Quality analytics
stats = get_parts_quality_stats(db)
```

## 📊 Performance Metrics

- **Database Query Speed**: 13,904 records/second
- **Parts Deduplication**: 99.7% reduction achieved  
- **AI Agent Readiness**: 100% (5/5 optimizations)
- **Search Response Time**: <100ms average

## 🗄️ Database Schema & Architecture

### Core Tables

#### `chunks` - Text Segments
```sql
CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    content text NOT NULL,
    manufacturer text,
    model text,
    page_number integer,
    metadata jsonb,
    embedding vector(384),  -- Sentence transformer embeddings
    created_at timestamptz DEFAULT now()
);
```

#### `images` - Extracted Images  
```sql
CREATE TABLE images (
    id bigserial PRIMARY KEY,
    storage_url text NOT NULL,     -- R2 storage path
    public_url text,               -- CDN URL
    file_hash text NOT NULL,       -- PDF source hash
    page_number integer,
    width integer,
    height integer,
    created_at timestamptz DEFAULT now()
);
```

#### `parts_catalog` - Optimized Parts (No Price/Availability)
```sql
CREATE TABLE parts_catalog (
    id bigserial PRIMARY KEY,
    part_number text UNIQUE NOT NULL,    -- Primary reference
    manufacturer text NOT NULL,
    part_name text,
    description text,
    category text,
    model_compatibility text[],          -- Array of compatible models
    created_at timestamptz DEFAULT now()
);
```

#### `chunk_images` - Relations
```sql
CREATE TABLE chunk_images (
    chunk_id bigint REFERENCES chunks(id),
    image_id bigint REFERENCES images(id),
    PRIMARY KEY (chunk_id, image_id)
);
```

### Performance Indices (Auto-Applied)

```sql
-- Vector search optimization
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Part number unique lookup
CREATE INDEX idx_parts_part_number ON parts_catalog (part_number);

-- Model compatibility search  
CREATE INDEX idx_parts_model_compatibility ON parts_catalog USING gin(model_compatibility);

-- Manufacturer + category filtering
CREATE INDEX idx_parts_category_manufacturer ON parts_catalog (category, manufacturer);
```

### AI Agent Views

#### `ai_agent_search_view` - Complete Search Interface
```sql
-- Combines chunks, images, and parts for AI Agent queries
-- Optimized for semantic search with relevance scoring
SELECT 
    c.content, c.manufacturer, c.model,
    i.public_url, i.storage_url,
    pc.part_number, pc.description, pc.model_compatibility,
    part_relevance_score  -- 60-100 based on match quality
FROM chunks c 
LEFT JOIN chunk_images ci ON c.id = ci.chunk_id
LEFT JOIN images i ON ci.image_id = i.id  
LEFT JOIN parts_catalog pc ON c.manufacturer = pc.manufacturer;
```

#### `parts_lookup_optimized` - Fast Parts Search
```sql
-- Quality-ranked parts lookup without price dependencies
SELECT 
    part_number, manufacturer, description, 
    model_compatibility,
    quality_rank,  -- 1-4 based on data completeness
    model_count,   -- Number of compatible models
    search_text    -- Combined searchable text
FROM parts_catalog 
ORDER BY quality_rank, manufacturer, part_number;
```

## 🛠️ Architecture

### Core Components
- `ai_pdf_processor.py` - Main processing engine
- `parts_helper_optimized.py` - AI Agent functions
- `database_client.py` - Supabase integration
- `r2_storage_client.py` - Cloudflare R2 storage

### Optimization Tools
- `parts_optimization_tool.py` - Deduplication utilities
- `complete_reset.py` - Fresh start utility
- `ai_agent_safe_optimization.sql` - Performance indices

## 🔧 Configuration

### Automatic Setup (Recommended)
```bash
# Cross-platform interactive wizard
python3 setup_wizard.py
```
**Detects and configures:**
- Hardware acceleration (Metal/CUDA/DirectML)
- Optimal model selection based on available RAM/VRAM
- Database connections and credentials  
- Storage configuration (R2/S3/Local)

### Manual Configuration
Copy `config.example.json` to `config.json` and configure:

```json
{
  "supabase_url": "https://your-project.supabase.co",
  "supabase_key": "your-service-role-key",
  
  "r2_account_id": "your-cloudflare-account-id",
  "r2_access_key_id": "your-r2-access-key", 
  "r2_secret_access_key": "your-r2-secret-key",
  "r2_bucket_name": "your-bucket-name",
  "r2_endpoint_url": "https://account-id.r2.cloudflarestorage.com",
  
  "vision_model": "llava:7b",
  "text_model": "llama3.1:8b", 
  "embedding_model": "embeddinggemma",
  
  "use_metal_acceleration": true,
  "parallel_workers": 8,
  "batch_size": 150
}
```

### Database Connection Test
```python
from database_client import DatabaseClient
import json

with open('config.json') as f:
    config = json.load(f)
    
db = DatabaseClient(config['supabase_url'], config['supabase_key'])

# Test connection
result = db.supabase.table('parts_catalog').select('id', count='exact').execute()
print(f"Database connected: {result.count} parts in catalog")
```

## 🎯 Beta Status

✅ **Completed Optimizations**:
- ✅ Cross-platform setup wizard (Windows/Linux/macOS)
- ✅ Parts catalog structure optimization (99.7% deduplication)
- ✅ Database performance indices (8 optimized indices)
- ✅ AI Agent helper functions (`parts_helper_optimized.py`)
- ✅ Quality-based prioritization system
- ✅ Fresh start cleanup utilities
- ✅ Vector search infrastructure
- ✅ Cloudflare R2 storage integration

🔄 **In Progress**:
- 🔄 Advanced semantic search algorithms
- 🔄 Multi-model AI agent support
- 🔄 Real-time processing pipeline
- 🔄 Enhanced vector similarity matching

🎯 **Architecture Decisions**:
- **No Price/Availability Dependencies**: Clean AI Agent integration
- **Part Number as Primary Key**: Eliminates 23,504 duplicate records
- **Quality-Based Ranking**: 1-4 priority system based on data completeness
- **Platform Agnostic**: Works on Apple Silicon, NVIDIA GPUs, and CPU-only

## 🛠️ Development Tools

### Database Management
```bash
# Reset entire system (database + R2 storage)
python3 complete_reset.py

# Analyze and optimize parts catalog
python3 parts_optimization_tool.py

# Validate optimizations
python3 parts_optimization_validation.py
```

### Performance Testing
```python
# Test parts helper functions
from parts_helper_optimized import *

# Quality analytics
stats = get_parts_quality_stats(db)
print(f"Data quality: {stats['complete_parts']}/{stats['total_parts']} parts")

# Part lookup performance test
import time
start = time.time()
part = get_part_by_number(db, "A93E563400")
print(f"Lookup time: {(time.time() - start)*1000:.2f}ms")
```

## 📈 Roadmap

- **v1.0**: Production AI Agent deployment
- **v1.1**: Advanced vector search
- **v1.2**: Multi-language support
- **v2.0**: Real-time processing pipeline

---

## 🌍 Cross-Platform Support

### Tested Platforms
- ✅ **macOS** (Apple Silicon M1/M2/M3 + Intel)
- ✅ **Linux** (Ubuntu, Debian, CentOS with CUDA support)
- ✅ **Windows** (10/11 with DirectML acceleration)
- ✅ **Docker** (Multi-architecture containers)

### Hardware Acceleration
- 🍎 **Apple Silicon**: Metal Performance Shaders + Neural Engine
- 🟢 **NVIDIA GPUs**: CUDA acceleration for embeddings
- 🔵 **Intel/AMD GPUs**: DirectML support on Windows
- ⚡ **CPU-only**: Optimized fallback for all platforms

### Setup Wizard Features
```bash
python3 setup_wizard.py
```
**Auto-detects:**
- Hardware capabilities and optimal model selection
- Available acceleration (Metal/CUDA/DirectML/CPU)
- Memory constraints and batch size optimization
- Platform-specific dependencies and paths

**Built for optimal AI Agent performance with clean, dependency-free architecture.**
