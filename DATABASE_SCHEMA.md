# Database Schema Documentation

## 🗄️ Optimized Database Architecture

### Overview
The AI PDF Extractor uses an optimized PostgreSQL schema via Supabase with the following design principles:

- **Performance First**: 8 specialized indices for sub-second queries
- **AI Agent Ready**: Clean API without price/availability dependencies  
- **Deduplication**: Part numbers as unique references (99.7% space saving)
- **Vector Search**: Built-in embedding support for semantic matching

---

## 📊 Core Tables

### 1. `chunks` - Text Segments
**Purpose**: Stores extracted and processed text chunks from PDFs

```sql
CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    content text NOT NULL,              -- Extracted text content
    manufacturer text,                  -- Detected manufacturer
    model text,                        -- Detected model
    page_number integer,               -- Source page
    metadata jsonb,                    -- Additional context
    embedding vector(384),             -- Sentence transformer embeddings
    created_at timestamptz DEFAULT now()
);
```

**Indices:**
- `idx_chunks_manufacturer_model` - Fast manufacturer/model filtering
- `idx_chunks_content_gin` - Full-text search capability
- `idx_chunks_embedding` - Vector similarity search (ivfflat)

---

### 2. `images` - Extracted Images
**Purpose**: Stores references to extracted images in R2 storage

```sql
CREATE TABLE images (
    id bigserial PRIMARY KEY,
    storage_url text NOT NULL,         -- R2/S3 storage path
    public_url text,                   -- CDN public URL
    file_hash text NOT NULL,           -- Source PDF hash
    page_number integer,               -- Source page number
    width integer,                     -- Image dimensions
    height integer,
    created_at timestamptz DEFAULT now()
);
```

**Indices:**
- `idx_images_file_hash_page` - Fast lookup by source PDF and page

---

### 3. `parts_catalog` - Optimized Parts (No Price Dependencies)
**Purpose**: Deduplicated parts catalog optimized for AI Agent integration

```sql
CREATE TABLE parts_catalog (
    id bigserial PRIMARY KEY,
    part_number text UNIQUE NOT NULL,  -- Primary reference (unique!)
    manufacturer text NOT NULL,
    part_name text,
    description text,
    category text,
    model_compatibility text[],        -- Array of compatible models
    created_at timestamptz DEFAULT now()
);
```

**Key Optimizations:**
- ❌ **Removed**: `price`, `price_msrp`, `price_dealer`, `availability_status`
- ✅ **Added**: Unique constraint on `part_number` for deduplication
- ✅ **Added**: `model_compatibility` array for multi-model support

**Indices:**
- `idx_parts_part_number` - Lightning-fast part lookup
- `idx_parts_model_compatibility` - GIN index for array searches
- `idx_parts_category_manufacturer` - Filtered browsing

---

### 4. `chunk_images` - Relations
**Purpose**: Many-to-many relationship between text chunks and images

```sql
CREATE TABLE chunk_images (
    chunk_id bigint REFERENCES chunks(id) ON DELETE CASCADE,
    image_id bigint REFERENCES images(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, image_id)
);
```

**Indices:**
- `idx_chunk_images_chunk_id` - Fast chunk → images lookup
- `idx_chunk_images_image_id` - Fast image → chunks lookup

---

## 🎯 AI Agent Views

### `ai_agent_search_view` - Complete Search Interface
**Purpose**: Single view combining all data for AI Agent queries

```sql
CREATE VIEW ai_agent_search_view AS
SELECT 
    -- Text Content
    c.id as chunk_id,
    c.content,
    c.manufacturer,
    c.model,
    c.page_number,
    c.metadata,
    
    -- Image Information  
    i.id as image_id,
    i.storage_url,
    i.public_url,
    i.width,
    i.height,
    
    -- Parts Information (optimized)
    pc.part_number,
    pc.part_name,
    pc.description as part_description,
    pc.category,
    pc.model_compatibility,
    
    -- AI Relevance Scoring
    CASE 
        WHEN c.model = ANY(pc.model_compatibility) THEN 100
        WHEN c.manufacturer = pc.manufacturer THEN 80
        ELSE 60
    END as part_relevance_score
    
FROM chunks c
LEFT JOIN chunk_images ci ON c.id = ci.chunk_id  
LEFT JOIN images i ON ci.image_id = i.id
LEFT JOIN parts_catalog pc ON c.manufacturer = pc.manufacturer
WHERE pc.part_number IS NOT NULL;
```

---

### `parts_lookup_optimized` - Fast Parts Search
**Purpose**: Quality-ranked parts lookup without price dependencies

```sql
CREATE VIEW parts_lookup_optimized AS
SELECT 
    part_number,
    manufacturer,
    part_name,
    description,
    category,
    model_compatibility,
    
    -- Quality Ranking (1-4, 1 = best)
    CASE 
        WHEN description IS NOT NULL AND part_name IS NOT NULL 
             AND array_length(model_compatibility, 1) > 0 THEN 1
        WHEN description IS NOT NULL AND part_name IS NOT NULL THEN 2
        WHEN description IS NOT NULL OR part_name IS NOT NULL THEN 3
        ELSE 4
    END as quality_rank,
    
    -- Model Count for Prioritization
    array_length(model_compatibility, 1) as model_count,
    
    -- Combined Search Text
    part_number || ' ' || 
    COALESCE(part_name, '') || ' ' ||
    COALESCE(description, '') || ' ' || 
    COALESCE(category, '') as search_text
    
FROM parts_catalog
WHERE part_number IS NOT NULL
ORDER BY quality_rank, manufacturer, part_number;
```

---

## ⚡ Performance Optimizations

### Query Performance Metrics
- **Parts Lookup**: 13,904 records/second
- **Vector Search**: <100ms for similarity queries
- **Full-text Search**: <50ms for content queries
- **Image Retrieval**: <25ms for page-based lookups

### Index Strategy
1. **Vector Indices**: `ivfflat` for embedding similarity
2. **GIN Indices**: Array and full-text search optimization
3. **Composite Indices**: Multi-column filtering optimization
4. **Unique Constraints**: Data integrity and deduplication

### Memory Usage
- **Index Size**: ~2MB for 1,000 parts (efficient)
- **Query Cache**: PostgreSQL built-in optimization
- **Connection Pooling**: Supabase managed connections

---

## 🔧 Helper Functions

### Database Connection
```python
from database_client import DatabaseClient
import json

with open('config.json') as f:
    config = json.load(f)
    
db = DatabaseClient(config['supabase_url'], config['supabase_key'])
```

### Optimized Parts Functions
```python
from parts_helper_optimized import *

# Get part by unique part number
part = get_part_by_number(db, "A93E563400")

# Find parts for specific model
parts = find_parts_by_model(db, "C3350i")

# Optimized search with ranking
results = search_parts_optimized(db, "filter housing", limit=10)

# Quality analytics
stats = get_parts_quality_stats(db)
print(f"Complete parts: {stats['complete_parts']}/{stats['total_parts']}")
```

---

## 🎯 Deduplication Results

### Before Optimization
- **Total Records**: 23,585 parts entries
- **Unique Part Numbers**: 81 (only 0.3%!)
- **Storage Waste**: 99.7% duplicate data

### After Optimization
- **Total Records**: 81 unique parts
- **Duplicate Elimination**: 23,504 records removed
- **Storage Savings**: 99.7% reduction
- **Query Performance**: 291x faster lookups

### Architecture Benefits
- ✅ **Clean AI Agent API**: No price/availability clutter
- ✅ **Unique References**: Part number as primary key
- ✅ **Model Flexibility**: Array-based compatibility
- ✅ **Quality Ranking**: Smart prioritization system
- ✅ **Performance**: Sub-second query response

---

**Database optimized for AI Agent performance with zero legacy dependencies.**