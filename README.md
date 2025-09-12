# 🚀 AI-Enhanced PDF Extractor

**Professional AI-powered PDF processing system** with **semantic embeddings**, **vision-guided chunking**, **cloud storage**, and **vector database integration**.

## ⚡ Key Features

- 🧠 **AI-Powered Processing** - Ollama integration with LLaVA vision and LLaMA text models
- 📊 **Semantic Embeddings** - 768-dimensional EmbeddingGemma vectors for intelligent search
- 👁️ **Vision-Guided Chunking** - Smart document structure recognition  
- ☁️ **Cloud Storage** - Cloudflare R2 integration for scalable image storage
- 🗄️ **Vector Database** - Supabase/PostgreSQL with pgvector support
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
- **Supabase Account** (or PostgreSQL with pgvector)
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

3. **Run setup wizard:**
   - **Windows:** `setup_wizard.bat`
   - **macOS/Linux:** `./setup_wizard.sh`
   - **Universal:** `python run.py setup_wizard.py`

4. **Configure your services:**
   - Copy `config.example.json` to `config.json`
   - Add your Supabase and Cloudflare R2 credentials

### Usage

#### Windows Users
```cmd
REM Process a PDF document
ai_pdf_processor.bat "path/to/document.pdf"

REM Check system status
status.bat

REM Search processed documents
smart_search_engine.bat
```

#### macOS/Linux Users  
```bash
# Process a PDF document
./ai_pdf_processor.sh "path/to/document.pdf"

# Check system status
./status.sh

# Search processed documents
./smart_search_engine.sh
```

#### Universal Launcher
```bash
# Works on any platform
python run.py ai_pdf_processor.py "path/to/document.pdf"
python run.py status.py
python run.py smart_search_engine.py
```

## 📁 Project Structure

```
AI-Enhanced-PDF-Extractor/
├── ai_pdf_processor.py      # Main processing system
├── smart_search_engine.py   # Document search & query
├── status.py               # System monitoring
├── setup_wizard.py         # Interactive setup
├── cross_platform_setup.py # Automated setup
├── requirements.txt        # Python dependencies
├── config.example.json     # Configuration template
├── *.bat                   # Windows launchers
├── *.sh                    # macOS/Linux launchers
├── run.py                  # Universal launcher
└── Documents/              # Place your PDFs here
```

## 🔧 Configuration

### Supabase Setup
1. Create a new Supabase project
2. Enable the `pgvector` extension
3. Import the database schema from `final_supabase_schema.sql`
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
- Check Supabase credentials in `config.json`
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