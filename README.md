# 🚀 AI-Enhanced PDF Extractor

**Professional AI-powered PDF processing system** with **semantic embeddin### Database Setup (PostgreSQL with pgvector)
1. Create a PostgreSQL database with pgvector extension (Supabase, AWS RDS, or self-hosted)
2. Enable the pgvector extension: `CREATE EXTENSION vector;`
3. Import the database schema from `database_schema.sql`, **vision-guided chunking**, **cloud storage**, and **vector database integration**.

## ⚡ Key Features

- 🧠 **AI-Powered Processing** - Ollama integration with LLaVA vision and LLaMA text models
- 📊 **Semantic Embeddings** - 768-dimensional EmbeddingGemma vectors for intelligent search
- 👁️ **Vision-Guided Chunking** - Smart document structure recognition  
- ☁️ **Cloud Storage** - Cloudflare R2 integration for scalable image storage
- 🗄️ **Vector Database** - PostgreSQL with pgvector support (Supabase, AWS RDS, etc.)
- 🔄 **Smart Resume** - Automatic continuation of interrupted processing
- 📈 **Progress Tracking** - Real-time status monitoring and analytics

## 🏗️ System Architecture

```
PDF Document → AI Vision Analysis → Smart Chunking → Embedding Generation → Vector Storage
                     ↓
              Image Extraction → Cloud Upload → Database Relations → Search Index
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Windows: `python`, macOS/Linux: `python3`)
- **Ollama** with models: `llama3.1:8b`, `llava:7b`, `embeddinggemma`
- **PostgreSQL Database with pgvector** (Supabase, AWS RDS, or self-hosted)
- **Cloudflare R2** (or S3-compatible storage)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tobiashaas/AI-Enhanced-PDF-Extractor.git
   cd AI-Enhanced-PDF-Extractor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Quick start:**
   ```bash
   python launch.py setup    # Interactive setup
   python launch.py process "your-document.pdf"
   ```

4. **Configure your services:**
   - **Recommended**: Use `python launch.py setup` for guided configuration
   - **Manual Option A**: Use `config.template.json` for generic providers (AWS, GCP, Azure)
   - **Manual Option B**: Use `config.example.json` for Supabase/R2 setups
   - Copy your chosen template to `config.json` and customize
   - The setup wizard automatically selects providers and guides configuration

### Usage

#### Universal Launcher (All Platforms)
The easiest way to use the system on any platform:

```bash
# Interactive setup with provider selection
python launch.py setup

# Process a PDF document
python launch.py process "path/to/document.pdf"

# Check system status  
python launch.py status

# Search processed documents
python launch.py search

# Run setup wizard
python launch.py setup
```

#### Direct Python Execution
```bash
# Alternative: Direct execution
python ai_pdf_processor.py "path/to/document.pdf"
python status.py
python smart_search_engine.py
```

The universal launcher automatically:
- ✅ **Detects your OS** (Windows/macOS/Linux)
- ✅ **Finds Python** (`python` vs `python3`)
- ✅ **Detects hardware** (Apple Silicon/NVIDIA/CPU)
- ✅ **Optimizes settings** (workers, batch size, GPU acceleration)
- ✅ **Checks dependencies** (Ollama, config, packages)

## 📁 Project Structure

```
AI-Enhanced-PDF-Extractor/
├── launch.py               # 🌍 Universal launcher (auto-detects OS/hardware)
├── run                     # 🚀 Bootstrap script (finds Python automatically)
├── ai_pdf_processor.py     # Main processing system
├── smart_search_engine.py  # Document search & query
├── status.py              # System monitoring
├── setup_wizard.py        # Interactive setup
├── cross_platform_setup.py # Automated setup
├── requirements.txt       # Python dependencies
├── config.example.json    # Configuration template
└── Documents/             # Place your PDFs here
```

## 🔧 Configuration

### Database Setup (PostgreSQL with pgvector)
1. Create a PostgreSQL database with pgvector extension (Supabase, AWS RDS, or self-hosted)
2. Enable the pgvector extension: `CREATE EXTENSION vector;`
3. Import the database schema from `database_schema.sql`
4. Add your credentials to `config.json`

### Cloudflare R2 Setup
1. Create an R2 bucket
2. Generate API tokens
3. Configure public domain for image access
4. Add credentials to `config.json`

### Ollama Models
```bash
# Install required models
ollama pull llama3.1:8b
ollama pull llava:7b  
ollama pull embeddinggemma
```

## 🎯 Supported Document Types

- **Service Manuals** (HP, Canon, Xerox, Brother, etc.)
- **Technical Documentation**
- **User Guides** 
- **Installation Manuals**
- **Troubleshooting Guides**

## 📊 Features

### AI Processing
- **Intelligent Chunking** - Context-aware text segmentation
- **Image Extraction** - Automatic diagram and figure detection
- **Metadata Enhancement** - AI-powered document classification
- **Error Code Detection** - Automatic identification of diagnostic codes

### Storage & Search
- **Vector Embeddings** - Semantic similarity search
- **Image Storage** - Cloud-based with CDN delivery
- **Chunk Relations** - Smart linking of text and images
- **Full-Text Search** - PostgreSQL text search integration

### Performance
- **Batch Processing** - Handle large documents efficiently
- **Smart Resume** - Continue from interruption points
- **Duplicate Detection** - Avoid reprocessing existing content
- **Progress Tracking** - Real-time status updates

## 🛠️ Troubleshooting

### Common Issues

**Python command not found:**
- Windows: Install Python from python.org, use `python` command
- macOS: Use `python3` or install via Homebrew
- Linux: Install via package manager: `sudo apt install python3`

**Ollama connection issues:**
- Ensure Ollama is running: `ollama serve`
- Check models are installed: `ollama list`
- Verify localhost:11434 accessibility

**Database connection errors:**
- Check database credentials in `config.json`
- Verify database schema is imported
- Ensure `pgvector` extension is enabled

## 📈 Performance Tips

- **Large PDFs:** Use batch processing with `--batch-size` parameter
- **Hardware:** Enable GPU acceleration for faster AI processing
- **Storage:** Configure R2 public domain for faster image access
- **Memory:** Increase batch size for more RAM-efficient processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test cross-platform compatibility
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation:** Check the wiki for detailed guides
- **Issues:** Report bugs via GitHub Issues
- **Discussions:** Join the GitHub Discussions for questions

---

**Made with ❤️ for document processing automation**