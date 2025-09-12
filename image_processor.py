#!/usr/bin/env python3
"""
Image Processing Module
Handles vector graphics rendering and image extraction from PDFs
"""

import fitz
import hashlib
import logging
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from PIL import Image


class ImageProcessor:
    """Advanced image processing with vector and raster export support"""
    
    def __init__(self, render_dpi: int = 144, export_vectors: bool = True, export_raster: bool = True):
        self.render_dpi = render_dpi
        self.export_vectors = export_vectors  # SVG export for technical drawings
        self.export_raster = export_raster    # PNG export for viewing/thumbnails
        
    def extract_images_from_page(self, page: fitz.Page, file_hash: str, page_num: int) -> List[Dict]:
        """Extract all images from a PDF page with vector and raster support"""
        images = []
        
        # 1. Extract embedded raster images
        raster_images = self._extract_raster_images(page, file_hash, page_num)
        images.extend(raster_images)
        
        # 2. Export vector graphics (SVG for technical drawings)
        vector_images = []
        if self.export_vectors:
            vector_images = self._export_vector_graphics_as_svg(page, file_hash, page_num)
            images.extend(vector_images)
        
        # 3. Render vector graphics as raster images (PNG for viewing)
        raster_renders = []
        if self.export_raster:
            raster_renders = self._render_vector_graphics_as_images(page, file_hash, page_num)
            images.extend(raster_renders)
        
        logging.info(f"📄 Page {page_num}: {len(raster_images)} raster, {len(vector_images) if self.export_vectors else 0} vector SVG, {len(raster_renders) if self.export_raster else 0} raster renders")
        return images
    
    def _extract_raster_images(self, page: fitz.Page, file_hash: str, page_num: int) -> List[Dict]:
        """Extract embedded raster images from PDF page"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Convert to PNG if necessary
                if pix.n - pix.alpha < 4:  # Not CMYK
                    image_data = pix.tobytes("png")
                else:  # CMYK - convert via PIL
                    pil_image = Image.frombytes("CMYK", [pix.width, pix.height], pix.samples)
                    rgb_image = pil_image.convert('RGB')
                    
                    buffer = BytesIO()
                    rgb_image.save(buffer, format='PNG')
                    image_data = buffer.getvalue()
                
                # Generate image hash
                image_hash = hashlib.sha256(image_data).hexdigest()
                
                images.append({
                    'image_data': image_data,
                    'metadata': {
                        'file_hash': file_hash,
                        'page_number': page_num,
                        'image_index': img_index,
                        'image_hash': image_hash,
                        'width': pix.width,
                        'height': pix.height,
                        'image_type': 'raster_embedded',
                        'extraction_method': 'embedded_image_extraction',
                        'file_size_bytes': len(image_data),
                        'image_format': 'png'
                    }
                })
                
                pix = None  # Free memory
                
            except Exception as e:
                logging.warning(f"⚠️ Failed to extract raster image {img_index} from page {page_num}: {e}")
                
        return images
    
    def _export_vector_graphics_as_svg(self, page: fitz.Page, file_hash: str, page_num: int) -> List[Dict]:
        """Export vector graphics from page as SVG for infinite zoom"""
        images = []
        
        try:
            # Check if page has vector content
            drawings = page.get_drawings()
            if not drawings:
                return images
            
            # Get page dimensions
            rect = page.rect
            
            # Skip if page is too small
            if rect.width < 50 or rect.height < 50:
                return images
            
            # Export page as SVG
            svg_data = page.get_svg_image(matrix=fitz.Identity)
            
            if svg_data:
                # Convert to bytes if needed
                if isinstance(svg_data, str):
                    svg_bytes = svg_data.encode('utf-8')
                else:
                    svg_bytes = svg_data
                
                # Generate metadata
                svg_hash = hashlib.sha256(svg_bytes).hexdigest()
                
                # Analyze vector content
                vector_analysis = self._analyze_vector_content(page)
                
                images.append({
                    'image_data': svg_bytes,
                    'metadata': {
                        'file_hash': file_hash,
                        'page_number': page_num,
                        'image_index': 0,  # SVG export is always index 0
                        'image_hash': svg_hash,
                        'width': int(rect.width),
                        'height': int(rect.height),
                        'image_type': 'vector_svg',
                        'extraction_method': 'svg_page_export',
                        'vector_path_count': vector_analysis['path_count'],
                        'has_text': vector_analysis['has_text'],
                        'has_images': vector_analysis['has_images'],
                        'file_size_bytes': len(svg_bytes),
                        'image_format': 'svg',
                        'scalable': True,  # Key advantage for technical drawings
                        'zoom_friendly': True
                    }
                })
                
                logging.info(f"📐 SVG Export Page {page_num}: {vector_analysis['path_count']} vector paths, {len(svg_bytes)} bytes")
                
        except Exception as e:
            logging.warning(f"⚠️ Failed to export SVG from page {page_num}: {e}")
            
        return images
    
    def _render_vector_graphics_as_images(self, page: fitz.Page, file_hash: str, page_num: int) -> List[Dict]:
        """Render vector graphics on page as raster images"""
        images = []
        
        try:
            # Get page dimensions
            rect = page.rect
            
            # Skip if page is too small
            if rect.width < 50 or rect.height < 50:
                return images
            
            # Calculate matrix for DPI scaling
            zoom_x = zoom_y = self.render_dpi / 72.0  # 72 DPI is default
            matrix = fitz.Matrix(zoom_x, zoom_y)
            
            # Render page as image
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PNG
            image_data = pix.tobytes("png")
            
            # Generate metadata
            image_hash = hashlib.sha256(image_data).hexdigest()
            
            # Analyze vector content
            vector_analysis = self._analyze_vector_content(page)
            
            images.append({
                'image_data': image_data,
                'metadata': {
                    'file_hash': file_hash,
                    'page_number': page_num,
                    'image_index': 1,  # Raster rendering is index 1 (after SVG)
                    'image_hash': image_hash,
                    'width': pix.width,
                    'height': pix.height,
                    'image_type': 'vector_raster_render',
                    'extraction_method': 'vector_page_rendering',
                    'render_resolution': f'{self.render_dpi}dpi',
                    'vector_path_count': vector_analysis.get('path_count', 0),
                    'has_text': vector_analysis.get('has_text', False),
                    'has_images': vector_analysis.get('has_images', False),
                    'file_size_bytes': len(image_data),
                    'image_format': 'png',
                    'scalable': False,  # Raster = not scalable
                    'zoom_friendly': False,
                    'purpose': 'thumbnail_viewing'  # Purpose clarification
                }
            })
            
            pix = None  # Free memory
            
        except Exception as e:
            logging.warning(f"⚠️ Failed to render vector graphics for page {page_num}: {e}")
            
        return images
    
    def _analyze_vector_content(self, page: fitz.Page) -> Dict:
        """Analyze vector content on page with performance optimization"""
        try:
            # Quick check first - get page rect for basic info
            rect = page.rect
            
            # Fast vector analysis - avoid expensive get_drawings() for complex pages
            try:
                # Try a quick method first - check if page has vector content
                # by testing if page can be converted to SVG efficiently
                svg_test = page.get_svg_image(matrix=fitz.Identity)
                has_vectors = svg_test is not None and len(svg_test) > 1000  # Basic SVG has ~1000 chars
                
                if has_vectors:
                    # Estimate path count based on SVG size for performance
                    svg_length = len(svg_test) if isinstance(svg_test, str) else len(str(svg_test))
                    # Rough estimate: 1 path ≈ 50-100 chars in SVG
                    estimated_paths = min(svg_length // 75, 1000)  # Cap at 1000 for performance
                    path_count = estimated_paths
                else:
                    path_count = 0
                    
            except Exception:
                # Fallback - assume moderate complexity
                path_count = 50
                
            # Quick text check
            try:
                text_dict = page.get_text("dict")
                has_text = bool(text_dict.get("blocks", []))
            except:
                has_text = False
                
            # Quick image check  
            try:
                image_list = page.get_images()
                has_images = len(image_list) > 0
            except:
                has_images = False
            
            return {
                'path_count': path_count,
                'has_text': has_text,
                'has_images': has_images,
                'page_width': rect.width,
                'page_height': rect.height
            }
            
        except Exception as e:
            logging.warning(f"⚠️ Vector content analysis failed: {e}")
            return {
                'path_count': 0,
                'has_text': False,
                'has_images': False,
                'page_width': 0,
                'page_height': 0
            }
    
    def resize_image(self, image_data: bytes, max_width: int = 1024, max_height: int = 1024) -> bytes:
        """Resize image while maintaining aspect ratio"""
        try:
            with Image.open(BytesIO(image_data)) as img:
                # Calculate new size
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Save to buffer
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                return buffer.getvalue()
                
        except Exception as e:
            logging.warning(f"⚠️ Failed to resize image: {e}")
            return image_data
    
    def generate_thumbnail(self, image_data: bytes, size: Tuple[int, int] = (200, 200)) -> bytes:
        """Generate thumbnail from image data"""
        try:
            with Image.open(BytesIO(image_data)) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                return buffer.getvalue()
                
        except Exception as e:
            logging.warning(f"⚠️ Failed to generate thumbnail: {e}")
            return image_data