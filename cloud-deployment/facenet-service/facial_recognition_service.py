#!/usr/bin/env python3
"""
FaceNet Service Implementation
Facial recognition service using TensorFlow/Keras
"""

import os
import time
import pickle
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import redis
import json
import httpx
from supabase import create_client, Client

class FaceNetService:
    """FaceNet service for facial recognition"""
    
    def __init__(self):
        self.model = None
        self.face_embeddings = {}
        self.face_database = {}
        self.model_path = os.getenv("MODEL_PATH", "/app/models/facenet_mobile.h5")
        self.database_path = os.getenv("DATABASE_PATH", "/app/models/face_embeddings.pkl")
        self.confidence_threshold = 0.8
        self.similarity_threshold = 0.6
        
        # Supabase configuration
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_client = None
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the FaceNet service"""
        try:
            # Initialize Supabase client
            if self.supabase_url and self.supabase_key:
                self.supabase_client = create_client(self.supabase_url, self.supabase_key)
                self.logger.info("Supabase client initialized")
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Download model from Supabase Storage if not exists locally
            if not os.path.exists(self.model_path):
                self._download_model_from_supabase()
            
            # Load model
            if os.path.exists(self.model_path):
                self.logger.info(f"Loading FaceNet model from {self.model_path}")
                self.model = load_model(self.model_path)
            else:
                self.logger.info("Creating new FaceNet model")
                self.model = self._create_facenet_model()
                self.model.save(self.model_path)
                # Upload to Supabase Storage
                self._upload_model_to_supabase()
            
            # Load face database
            if os.path.exists(self.database_path):
                self.logger.info(f"Loading face database from {self.database_path}")
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.face_embeddings = data.get('embeddings', {})
                    self.face_database = data.get('database', {})
            else:
                self.logger.info("Creating new face database")
                self.face_embeddings = {}
                self.face_database = {}
                
            self.logger.info(f"FaceNet service initialized with {len(self.face_database)} faces")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FaceNet service: {e}")
            raise
    
    def _create_facenet_model(self):
        """Create a simple FaceNet-like model for demonstration"""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        # Use MobileNetV2 as base
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='linear')(x)  # Face embedding layer
        
        model = Model(inputs=base_model.input, outputs=x)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        return model
    
    def _download_model_from_supabase(self):
        """Download model from Supabase Storage"""
        if not self.supabase_client:
            self.logger.warning("Supabase client not initialized, skipping model download")
            return
        
        try:
            self.logger.info("Downloading FaceNet model from Supabase Storage...")
            
            # Download model file
            model_files = [
                "autosos/models/facenet/facenet_mobile.h5",
                "autosos/models/facenet/facenet_mobile.tflite"
            ]
            
            for model_file in model_files:
                try:
                    response = self.supabase_client.storage.from_("autosos").download(model_file)
                    if response:
                        # Save to local path
                        local_path = os.path.join("/app/models", os.path.basename(model_file))
                        with open(local_path, 'wb') as f:
                            f.write(response)
                        self.logger.info(f"Downloaded {model_file} to {local_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to download {model_file}: {e}")
            
            # Download face database if exists
            try:
                db_response = self.supabase_client.storage.from_("autosos").download("autosos/models/facenet/face_embeddings.pkl")
                if db_response:
                    with open(self.database_path, 'wb') as f:
                        f.write(db_response)
                    self.logger.info("Downloaded face database from Supabase Storage")
            except Exception as e:
                self.logger.warning(f"Face database not found in Supabase Storage: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to download models from Supabase: {e}")
    
    def _upload_model_to_supabase(self):
        """Upload model to Supabase Storage"""
        if not self.supabase_client:
            self.logger.warning("Supabase client not initialized, skipping model upload")
            return
        
        try:
            self.logger.info("Uploading FaceNet model to Supabase Storage...")
            
            # Upload model file
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = f.read()
                
                self.supabase_client.storage.from_("autosos").upload(
                    "autosos/models/facenet/facenet_mobile.h5",
                    model_data,
                    {"content-type": "application/octet-stream"}
                )
                self.logger.info("Uploaded FaceNet model to Supabase Storage")
            
            # Upload face database
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    db_data = f.read()
                
                self.supabase_client.storage.from_("autosos").upload(
                    "autosos/models/facenet/face_embeddings.pkl",
                    db_data,
                    {"content-type": "application/octet-stream"}
                )
                self.logger.info("Uploaded face database to Supabase Storage")
                
        except Exception as e:
            self.logger.error(f"Failed to upload models to Supabase: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for FaceNet model"""
        # Resize to model input size
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def _extract_face_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Get embedding
            embedding = self.model.predict(processed_image, verbose=0)
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.flatten()
            
        except Exception as e:
            self.logger.error(f"Failed to extract face embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def register_face(self, image: np.ndarray, user_id: str, user_name: str) -> Dict[str, Any]:
        """Register a new face"""
        try:
            # Extract face embedding
            embedding = self._extract_face_embedding(image)
            
            if embedding is None:
                return {
                    "success": False,
                    "error": "Failed to extract face embedding"
                }
            
            # Store embedding and user info
            self.face_embeddings[user_id] = embedding
            self.face_database[user_id] = {
                "user_name": user_name,
                "registered_at": time.time(),
                "embedding_shape": embedding.shape
            }
            
            # Save to disk
            self._save_database()
            
            self.logger.info(f"Face registered for user {user_id} ({user_name})")
            
            return {
                "success": True,
                "user_id": user_id,
                "user_name": user_name,
                "embedding_shape": embedding.shape,
                "message": "Face registered successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to register face: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def recognize_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Recognize a face from image"""
        try:
            # Extract face embedding
            query_embedding = self._extract_face_embedding(image)
            
            if query_embedding is None:
                return None
            
            # Find best match
            best_match = None
            best_similarity = 0.0
            
            for user_id, stored_embedding in self.face_embeddings.items():
                similarity = self._calculate_similarity(query_embedding, stored_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = user_id
            
            if best_match and best_similarity >= self.confidence_threshold:
                user_info = self.face_database[best_match]
                
                return {
                    "user_id": best_match,
                    "user_name": user_info["user_name"],
                    "confidence": best_similarity,
                    "similarity": best_similarity,
                    "recognized_at": time.time()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to recognize face: {e}")
            return None
    
    def get_face_count(self) -> int:
        """Get the number of registered faces"""
        return len(self.face_database)
    
    def _save_database(self):
        """Save face database to disk"""
        try:
            data = {
                'embeddings': self.face_embeddings,
                'database': self.face_database
            }
            
            with open(self.database_path, 'wb') as f:
                pickle.dump(data, f)
                
            self.logger.info("Face database saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save face database: {e}")
    
    def delete_face(self, user_id: str) -> Dict[str, Any]:
        """Delete a registered face"""
        try:
            if user_id in self.face_embeddings:
                del self.face_embeddings[user_id]
                del self.face_database[user_id]
                
                # Save to disk
                self._save_database()
                
                self.logger.info(f"Face deleted for user {user_id}")
                
                return {
                    "success": True,
                    "message": f"Face deleted for user {user_id}"
                }
            else:
                return {
                    "success": False,
                    "error": f"User {user_id} not found"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to delete face: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_faces(self) -> Dict[str, Any]:
        """List all registered faces"""
        try:
            faces = []
            for user_id, user_info in self.face_database.items():
                faces.append({
                    "user_id": user_id,
                    "user_name": user_info["user_name"],
                    "registered_at": user_info["registered_at"]
                })
            
            return {
                "success": True,
                "faces": faces,
                "count": len(faces)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list faces: {e}")
            return {
                "success": False,
                "error": str(e)
            }
