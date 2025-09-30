#!/usr/bin/env python3
"""
Script to check what models are in Supabase Storage
"""

import os
from supabase import create_client, Client

def check_supabase_models():
    """Check what models are in Supabase Storage"""
    
    # Supabase configuration
    supabase_url = "https://atdibhoeaeqfgjswcqwx.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF0ZGliaG9lYWVxZmdqc3djcXd4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzM1NTIzOSwiZXhwIjoyMDcyOTMxMjM5fQ.nJoAQZAcR7VeX-lmbKbtjHTjj5U5gfpavJ8fhgWTPU8"
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print("🔍 Checking Supabase Storage for models...")
    print(f"📡 Supabase URL: {supabase_url}")
    print(f"🪣 Bucket: autosos")
    
    try:
        # List all files in the autosos bucket
        print("\n📁 All files in 'autosos' bucket:")
        response = supabase.storage.from_("autosos").list()
        
        if response:
            for item in response:
                if item.get('name'):
                    print(f"   📄 {item['name']}")
                    if 'metadata' in item and 'size' in item['metadata']:
                        size_mb = item['metadata']['size'] / (1024 * 1024)
                        print(f"      Size: {size_mb:.2f} MB")
        else:
            print("   📭 No files found in bucket")
            
        # Check specifically for YOLOv8 models
        print("\n🎯 Looking for YOLOv8 models:")
        yolo_paths = [
            "autosos/models/yolov8/",
            "models/yolov8/",
            "yolov8/",
            "best.pt",
            "motorcycle_diagnostic.pt"
        ]
        
        for path in yolo_paths:
            try:
                files = supabase.storage.from_("autosos").list(path)
                if files:
                    print(f"   ✅ Found in '{path}':")
                    for file in files:
                        if file.get('name'):
                            print(f"      📄 {file['name']}")
                else:
                    print(f"   ❌ Nothing found in '{path}'")
            except Exception as e:
                print(f"   ❌ Error checking '{path}': {e}")
                
    except Exception as e:
        print(f"❌ Error accessing Supabase Storage: {e}")

if __name__ == "__main__":
    check_supabase_models()
