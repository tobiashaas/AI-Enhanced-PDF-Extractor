# Video Processing Module

This module handles video tutorial metadata extraction and processing through:
1. CSV imports (primary method)
2. Automated web scraping (optional feature)

## Key Features
- CSV import for video metadata
- Optional web scraping for video tutorials
- Integration with service manual content
- Manufacturer and model classification
- Semantic search via embeddings

## CSV Format

The CSV file should contain the following columns:

```csv
video_url,manufacturer,model,procedure_name,tutorial_type,difficulty_level,duration_minutes,language,quality_rating,tools_shown,parts_demonstrated,key_steps,common_mistakes
```

### Field Descriptions:

- **video_url:** Direct URL to the video
- **manufacturer:** Manufacturer name (e.g., "Lexmark")
- **model:** Exact model (e.g., "CX963se")
- **procedure_name:** Descriptive title
- **tutorial_type:** maintenance, repair, setup, troubleshooting
- **difficulty_level:** beginner, intermediate, advanced
- **duration_minutes:** Length in minutes
- **language:** Language code (de, en, etc.)
- **quality_rating:** 1-5 (5 = best)
- **tools_shown:** Tools separated by |
- **parts_demonstrated:** Part numbers separated by |
- **key_steps:** Main steps separated by |
- **common_mistakes:** Common errors separated by |

## Classes
- `VideoProcessor`: Main class for video data processing

## Usage Examples

### CSV Import:
```python
from modules.video_processing.processor import VideoProcessor

processor = VideoProcessor()
processor.process_csv("Documents/Video_Tutorials/demo_videos.csv")
```

### Automated Scraping (optional):
```python
processor.scrape_lexmark_videos("CX963")
```

## Setup for Scraping

1. Create `.env` file:
```env
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
```

2. First login:
- Launches browser
- Logs in (MS365)
- Saves auth in `auth.json`

## Integration

This module integrates with:
- Processing pipeline (for orchestration)
- Database (video_tutorials table)

### Database Schema

Videos are stored in the `video_tutorials` table with:
- Embeddings for semantic search
- Metadata for source/import tracking
- Arrays for tools/parts/steps