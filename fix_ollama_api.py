#!/usr/bin/env python3
"""
Ollama API Compatibility Fixer
==============================
Testet und behebt Ollama API-Endpunkt Probleme (Windows/macOS Unterschiede)
"""

import requests
import json
import sys

def test_ollama_endpoints():
    print("🔧 OLLAMA API COMPATIBILITY TEST")
    print("=" * 50)
    
    base_url = "http://localhost:11434"
    
    # Test basic connectivity
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            models = response.json().get('models', [])
            print(f"📊 Available models: {len(models)}")
        else:
            print("❌ Ollama service not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False
    
    # Test embedding endpoint (new API only since embeddinggemma requires Ollama >= 0.11.10)
    test_payload = {
        "model": "embeddinggemma",
        "prompt": "test embedding"
    }
    
    endpoint = "/api/embed"
    description = "Ollama API (requires v0.11.10+)"
    
    print(f"\n🧪 Testing {description}...")
    print(f"   URL: {base_url}{endpoint}")
    
    try:
        response = requests.post(
            f"{base_url}{endpoint}",
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            if embedding and len(embedding) > 0:
                print(f"   ✅ SUCCESS: Got {len(embedding)}-dimensional embedding")
                working_endpoint = endpoint
            else:
                print("   ⚠️  Response OK but empty embedding")
        elif response.status_code == 404:
            print(f"   ❌ 404 Not Found - Ollama version too old!")
            print(f"   💡 Please update Ollama to >= 0.11.10")
            print(f"   🔧 Run: curl -fsSL https://ollama.ai/install.sh | sh")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    if working_endpoint:
        print(f"\n🎉 SOLUTION FOUND!")
        print(f"✅ Working endpoint: {working_endpoint}")
        print(f"💡 Your AI PDF Processor will use this endpoint")
        return True
    else:
        print(f"\n❌ EMBEDDING API NOT WORKING!")
        print(f"🔧 Required: Ollama >= 0.11.10 with embeddinggemma model")
        print(f"🔧 Troubleshooting steps:")
        print(f"   1. Update Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print(f"   2. Restart Ollama service")
        print(f"   3. Install model: ollama pull embeddinggemma")
        return False

def check_ollama_version():
    """Check Ollama version for compatibility info"""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"\n📋 Ollama Version Info:")
            print(f"   Version: {version_info.get('version', 'Unknown')}")
            return version_info
    except:
        print(f"\n⚠️  Could not get Ollama version info")
    return None

if __name__ == "__main__":
    print("Starting Ollama API compatibility test...")
    
    # Check version first
    check_ollama_version()
    
    # Test endpoints
    success = test_ollama_endpoints()
    
    if success:
        print(f"\n🚀 Your system is ready for AI PDF processing!")
    else:
        print(f"\n🛠️  Please fix Ollama setup before proceeding")
        sys.exit(1)