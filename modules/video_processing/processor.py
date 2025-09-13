#!/usr/bin/env python3
"""
Video Processing Module
----------------------
Handles video metadata extraction and storage for service videos.
Supports both automated scraping and manual CSV imports.
"""

import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from supabase import create_client
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

class VideoProcessor:
    def __init__(self):
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def process_csv(self, csv_path: str) -> None:
        """Process videos from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                video_data = self._parse_csv_row(row)
                self._store_video(video_data)
                logger.info(f"Processed video: {video_data['procedure_name']}")
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
    
    def scrape_lexmark_videos(self, model: str) -> None:
        """Scrape videos from Lexmark support portal."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            try:
                context = browser.new_context(storage_state="auth.json")
                page = context.new_page()
                self._handle_lexmark_login(page)
                videos = self._scrape_model_page(page, model)
                for video in videos:
                    self._store_video(video)
            except Exception as e:
                logger.error(f"Error scraping videos: {e}")
            finally:
                browser.close()
    
    def _parse_csv_row(self, row: pd.Series) -> Dict:
        """Parse CSV row into video data structure."""
        return {
            'video_url': row['video_url'],
            'manufacturer': row['manufacturer'],
            'model': row['model'],
            'procedure_name': row['procedure_name'],
            'tutorial_type': row.get('tutorial_type', 'maintenance'),
            'difficulty_level': row.get('difficulty_level', 'intermediate'),
            'duration_minutes': row.get('duration_minutes', 0),
            'language': row.get('language', 'en'),
            'quality_rating': row.get('quality_rating', 5),
            'tools_shown': row.get('tools_shown', '').split('|') if row.get('tools_shown') else [],
            'parts_demonstrated': row.get('parts_demonstrated', '').split('|') if row.get('parts_demonstrated') else [],
            'key_steps': row.get('key_steps', '').split('|') if row.get('key_steps') else [],
            'common_mistakes': row.get('common_mistakes', '').split('|') if row.get('common_mistakes') else [],
            'metadata': {
                'source': 'Manual CSV Import',
                'import_date': datetime.now().isoformat()
            }
        }
    
    def _store_video(self, video_data: Dict) -> None:
        """Store video data in Supabase."""
        try:
            self.supabase.table('video_tutorials').insert(video_data).execute()
            logger.info(f"Stored video: {video_data['procedure_name']}")
        except Exception as e:
            logger.error(f"Error storing video: {e}")

    def _handle_lexmark_login(self, page) -> None:
        """Handle Lexmark MS365 login process."""
        # Implementation for MS365 login...
        pass

    def _scrape_model_page(self, page, model: str) -> List[Dict]:
        """Scrape videos from model-specific page."""
        # Implementation for scraping...
        pass

if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Example usage:
    # 1. Process CSV:
    # processor.process_csv("Documents/Video_Tutorials/demo_videos.csv")
    
    # 2. Scrape Lexmark:
    # processor.scrape_lexmark_videos("CX963")