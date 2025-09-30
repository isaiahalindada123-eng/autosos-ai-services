#!/usr/bin/env python3
"""
Script to upload face database to Supabase Storage
"""

import os
import pickle
from supabase import create_client, Client

def create_and_upload_face_database():
    """Create and upload face database to Supabase Storage"""
    
    # Supabase configuration
    supabase_url = "https://atdibhoeaeqfgjswcqwx.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF0ZGliaG9lYWVxZmdqc3djcXd4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzM1NTIzOSwiZXhwIjoyMDcyOTMxMjM5fQ.nJoAQZAcR7VeX-lmbKbtjHTjj5U5gfpavJ8fhgWTPU8"
    
    # Initialize Supabase client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print("Creating face database for Supabase Storage...")
    
    # Create empty face database structure
    face_database = {
        'embeddings': {},  # Empty embeddings dictionary
        'database': {}     # Empty database dictionary
    }
    
    try:
        # Convert to pickle format
        db_data = pickle.dumps(face_database)
        
        # Upload to Supabase Storage
        response = supabase.storage.from_("autosos").upload(
            "autosos/models/facenet/face_embeddings.pkl",
            db_data,
            {"content-type": "application/octet-stream"}
        )
        
        print("SUCCESS: Face database uploaded successfully!")
        print("Path: autosos/models/facenet/face_embeddings.pkl")
        print("Database contains: 0 registered faces (empty database)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error uploading face database: {e}")
        return False

if __name__ == "__main__":
    create_and_upload_face_database()
