#!/usr/bin/env python3
"""
Script to upload AutoSOS models to Supabase Storage
Run this script to upload your local models to the cloud
"""

import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_models_to_supabase():
    """Upload models to Supabase Storage"""
    
    # Supabase configuration
    supabase_url = os.getenv("SUPABASE_URL", "https://atdibhoeaeqfgjswcqwx.supabase.co")
    supabase_key = os.getenv("SUPABASE_KEY", "sb_publishable_8zWSuqsDoSKDiWkz3Yd_eg_E7N1X7oj")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return False
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print("🚀 Starting model upload to Supabase Storage...")
    print(f"📡 Supabase URL: {supabase_url}")
    print(f"🪣 Bucket: autosos")
    
    # Define model files to upload
    models_to_upload = [
        # FaceNet models
        {
            "local_path": "models/facenet_mobile.h5",
            "storage_path": "autosos/models/facenet/facenet_mobile.h5",
            "description": "FaceNet Mobile Model (H5)"
        },
        {
            "local_path": "models/facenet_mobile.tflite", 
            "storage_path": "autosos/models/facenet/facenet_mobile.tflite",
            "description": "FaceNet Mobile Model (TFLite)"
        },
        {
            "local_path": "face_database/face_embeddings.pkl",
            "storage_path": "autosos/models/facenet/face_embeddings.pkl",
            "description": "Face Database"
        },
        
        # YOLOv8 models
        {
            "local_path": "yolo-motorcycle-diagnostic-training/runs/detect/train/weights/best.pt",
            "storage_path": "autosos/models/yolov8/best.pt",
            "description": "YOLOv8 Best Model"
        },
        {
            "local_path": "yolo-motorcycle-diagnostic-training/runs/detect/train/weights/last.pt",
            "storage_path": "autosos/models/yolov8/last.pt", 
            "description": "YOLOv8 Last Model"
        }
    ]
    
    success_count = 0
    total_count = len(models_to_upload)
    
    for model in models_to_upload:
        local_path = model["local_path"]
        storage_path = model["storage_path"]
        description = model["description"]
        
        print(f"\n📁 Processing: {description}")
        print(f"   Local: {local_path}")
        print(f"   Storage: {storage_path}")
        
        # Check if local file exists
        if not os.path.exists(local_path):
            print(f"   ⚠️  Warning: Local file not found, skipping...")
            continue
        
        try:
            # Read file
            with open(local_path, 'rb') as f:
                file_data = f.read()
            
            # Upload to Supabase Storage
            response = supabase.storage.from_("autosos").upload(
                storage_path,
                file_data,
                {"content-type": "application/octet-stream"}
            )
            
            print(f"   ✅ Successfully uploaded!")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error uploading: {e}")
    
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 All models uploaded successfully!")
        return True
    else:
        print(f"\n⚠️  {total_count - success_count} models failed to upload")
        return False

def list_uploaded_models():
    """List models currently in Supabase Storage"""
    
    # Supabase configuration
    supabase_url = os.getenv("SUPABASE_URL", "https://atdibhoeaeqfgjswcqwx.supabase.co")
    supabase_key = os.getenv("SUPABASE_KEY", "sb_publishable_8zWSuqsDoSKDiWkz3Yd_eg_E7N1X7oj")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print("📋 Listing models in Supabase Storage...")
    
    try:
        # List files in the models directory
        response = supabase.storage.from_("autosos").list("autosos/models")
        
        if response:
            print("\n📁 Models in Supabase Storage:")
            for item in response:
                if item.get('name'):
                    print(f"   📄 {item['name']}")
                    if 'metadata' in item and 'size' in item['metadata']:
                        size_mb = item['metadata']['size'] / (1024 * 1024)
                        print(f"      Size: {size_mb:.2f} MB")
        else:
            print("   📭 No models found in storage")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    print("🤖 AutoSOS Model Uploader")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_uploaded_models()
    else:
        success = upload_models_to_supabase()
        
        if success:
            print("\n🔍 You can now list uploaded models with:")
            print("   python upload-models-to-supabase.py list")
        else:
            print("\n💡 Make sure your model files exist in the correct paths:")
            print("   - models/facenet_mobile.h5")
            print("   - models/facenet_mobile.tflite") 
            print("   - face_database/face_embeddings.pkl")
            print("   - yolo-motorcycle-diagnostic-training/runs/detect/train/weights/best.pt")
