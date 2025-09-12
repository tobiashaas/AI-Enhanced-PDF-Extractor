#!/usr/bin/env python3
"""
Simple PDF Test ohne Database - nur Vision AI Test
"""

import fitz  # PyMuPDF
import requests
import json
import base64
from io import BytesIO
from PIL import Image

def test_pdf_extraction(pdf_path):
    """Teste nur die PDF-Extraktion ohne Database"""
    print(f"📄 Teste PDF-Extraktion: {pdf_path}")
    
    try:
        # PDF öffnen
        doc = fitz.open(pdf_path)
        print(f"✅ PDF geöffnet: {len(doc)} Seiten")
        
        # Erste Seite extrahieren
        page = doc[0]
        
        # Text extrahieren
        text = page.get_text()
        print(f"📝 Text auf Seite 1: {len(text)} Zeichen")
        if len(text) > 0:
            print(f"   Preview: {text[:200]}...")
        
        # Bild extrahieren
        pix = page.get_pixmap(dpi=144)
        img_data = pix.tobytes("png")
        print(f"🖼️  Bild extrahiert: {len(img_data)} bytes")
        
        # Test Vision AI (optional)
        try:
            print("🧠 Teste Vision AI...")
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llava:7b",
                    "prompt": "Describe what you see in this document briefly.",
                    "images": [img_base64],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                vision_result = result.get('response', 'No response')
                print(f"✅ Vision AI: {vision_result[:150]}...")
            else:
                print(f"❌ Vision AI failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  Vision AI test failed: {e}")
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"❌ PDF test failed: {e}")
        return False

def main():
    print("🧪 SIMPLE PDF PROCESSING TEST")
    print("=" * 50)
    print("Testing PDF extraction without database complications...")
    print()
    
    # Test mit einem kleinen PDF
    test_file = "Documents/Parts_Catalogs/Konica_Minolta/C451i/C451i_Parts.pdf"
    
    success = test_pdf_extraction(test_file)
    
    if success:
        print("\n✅ PDF PROCESSING GRUNDFUNKTIONEN ARBEITEN!")
        print("   Das Problem liegt im Database Schema Mismatch")
        print("   Nächster Schritt: Database Schema korrigieren")
    else:
        print("\n❌ PDF PROCESSING HAT GRUNDLEGENDE PROBLEME")

if __name__ == "__main__":
    main()