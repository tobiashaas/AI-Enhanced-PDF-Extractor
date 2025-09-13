# Video Tutorials

This directory contains video tutorials for printer repair and maintenance. These tutorials are categorized by manufacturer and model, providing visual guidance for technicians.

## Directory Structure

Videos are organized by manufacturer and then by printer model. Each video has associated metadata stored in CSV files.

## Metadata Format

Video metadata is stored in CSV format with the following structure:

- `video_id`: Unique identifier for the video
- `manufacturer`: Printer manufacturer name
- `model`: Printer model number
- `title`: Descriptive title of the video
- `description`: Detailed description of the video content
- `duration`: Length of the video in minutes and seconds
- `resolution`: Video resolution (e.g., 1080p, 720p)
- `file_path`: Path to the video file
- `url`: URL to the video if hosted online
- `tags`: Comma-separated list of relevant tags
- `created_date`: Date when the video was created
- `updated_date`: Date when the video was last updated
- `version`: Video version number

## CSV Files

- `demo_videos.csv`: Contains metadata for demonstration videos
- `video_metadata_template.csv`: Template for adding new video metadata

## Processing

Video metadata is processed by the video processing module, which extracts information from CSV files and makes it searchable through the AI system. The processor creates vector embeddings for semantic search capabilities.

## Adding New Videos

1. Add the video file to the appropriate manufacturer/model directory
2. Add the video metadata to the appropriate CSV file using the template format
3. Run the video processing module to update the database

## Accessing Videos

Videos can be accessed through:

1. Direct file system access using the file path
2. The AI search interface using natural language queries
3. API endpoints for integration with other systems

## Video Format Standards

- Preferred format: MP4 with H.264 encoding
- Maximum resolution: 1080p
- Audio: AAC encoding, stereo
- Frame rate: 30fps