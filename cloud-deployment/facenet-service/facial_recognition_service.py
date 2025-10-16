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
        self.model_path = os.getenv("MODEL_PATH", "autosos/models/facenet/facenet_mobile.h5")
        self.database_path = os.getenv("DATABASE_PATH", "autosos/models/facenet/face_embeddings.pkl")
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
            
            # Load model from Supabase Storage
            self._load_model_from_supabase()
            
            # Load face database from Supabase Storage
            self._load_database_from_supabase()
                
            self.logger.info(f"FaceNet service initialized with {len(self.face_database)} faces")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FaceNet service: {e}")
            raise
    
    def _load_model_from_supabase(self):
        """Load model from Supabase Storage"""
        try:
            if self.supabase_client:
                # Try to download model from Supabase Storage
                try:
                    model_response = self.supabase_client.storage.from_("autosos").download(self.model_path)
                    if model_response:
                        # Save to temporary local file for loading
                        temp_model_path = "/tmp/facenet_mobile.h5"
                        with open(temp_model_path, 'wb') as f:
                            f.write(model_response)
                        
                        self.logger.info(f"Loading FaceNet model from Supabase Storage: {self.model_path}")
                        self.model = load_model(temp_model_path)
                        self.logger.info("FaceNet model loaded successfully from Supabase")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to load model from Supabase Storage: {e}")
            
            # Fallback: create new model
            self.logger.info("Creating new FaceNet model")
            self.model = self._create_facenet_model()
            self._upload_model_to_supabase()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_database_from_supabase(self):
        """Load face database from Supabase Storage"""
        try:
            if self.supabase_client:
                # Try to download database from Supabase Storage
                try:
                    db_response = self.supabase_client.storage.from_("autosos").download(self.database_path)
                    if db_response:
                        data = pickle.loads(db_response)
                        self.face_embeddings = data.get('embeddings', {})
                        self.face_database = data.get('database', {})
                        
                        # Clean up corrupted embeddings
                        self._clean_corrupted_embeddings()
                        
                        self.logger.info(f"Loaded face database from Supabase Storage: {len(self.face_database)} faces")
                        return
                except Exception as e:
                    self.logger.warning(f"Failed to load database from Supabase Storage: {e}")
            
            # Fallback: create new database
            self.logger.info("Creating new face database")
            self.face_embeddings = {}
            self.face_database = {}
            
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            raise

    def _clean_corrupted_embeddings(self):
        """Clean up corrupted embeddings from the database"""
        try:
            corrupted_users = []
            
            for user_id, embedding in self.face_embeddings.items():
                # Check if embedding has correct shape (should be 128 dimensions)
                if embedding.shape != (128,) and embedding.shape != (1, 128):
                    self.logger.warning(f"Corrupted embedding for user {user_id}: shape {embedding.shape}")
                    corrupted_users.append(user_id)
                elif np.all(embedding == 0):
                    self.logger.warning(f"Zero embedding for user {user_id}")
                    corrupted_users.append(user_id)
            
            # Remove corrupted embeddings
            for user_id in corrupted_users:
                if user_id in self.face_embeddings:
                    del self.face_embeddings[user_id]
                if user_id in self.face_database:
                    del self.face_database[user_id]
                self.logger.info(f"Removed corrupted embedding for user {user_id}")
            
            if corrupted_users:
                self.logger.info(f"Cleaned up {len(corrupted_users)} corrupted embeddings")
                # Save cleaned database
                self._save_database()
                
        except Exception as e:
            self.logger.error(f"Failed to clean corrupted embeddings: {e}")
    
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
            
            # Save model to temporary file and upload
            temp_model_path = "/tmp/facenet_mobile.h5"
            self.model.save(temp_model_path)
            
            with open(temp_model_path, 'rb') as f:
                model_data = f.read()
            
            # Try to upload with upsert option first
            try:
                self.supabase_client.storage.from_("autosos").upload(
                    self.model_path,
                    model_data,
                    {"content-type": "application/octet-stream", "upsert": True}
                )
                self.logger.info(f"Uploaded FaceNet model to Supabase Storage: {self.model_path}")
            except Exception as upload_error:
                # If upsert fails, try delete then upload
                if "409" in str(upload_error) or "Duplicate" in str(upload_error):
                    self.logger.info("Model file exists, deleting and re-uploading...")
                    try:
                        # Delete existing file
                        self.supabase_client.storage.from_("autosos").remove([self.model_path])
                        # Upload new file
                        self.supabase_client.storage.from_("autosos").upload(
                            self.model_path,
                            model_data,
                            {"content-type": "application/octet-stream"}
                        )
                        self.logger.info(f"Updated FaceNet model in Supabase Storage: {self.model_path}")
                    except Exception as delete_error:
                        self.logger.error(f"Failed to delete and re-upload model: {delete_error}")
                        raise
                else:
                    raise upload_error
            
            # Clean up temporary file
            os.remove(temp_model_path)
                
        except Exception as e:
            self.logger.error(f"Failed to upload model to Supabase Storage: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for FaceNet model"""
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Expected numpy array, got {type(image)}")
                raise ValueError(f"Expected numpy array, got {type(image)}")
            
            if image.size == 0:
                self.logger.error("Image is empty")
                raise ValueError("Image is empty")
            
            self.logger.info(f"Preprocessing image with shape: {image.shape}, dtype: {image.dtype}")
            
            # Ensure image has 3 dimensions (height, width, channels)
            if len(image.shape) != 3:
                self.logger.error(f"Expected 3D image, got {len(image.shape)}D with shape {image.shape}")
                raise ValueError(f"Expected 3D image, got {len(image.shape)}D")
            
            # Resize to model input size
            image = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            self.logger.info(f"Preprocessed image shape: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _extract_face_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            # Validate input image
            if image is None or image.size == 0:
                self.logger.error("Invalid input image")
                return None
            
            self.logger.info(f"Input image shape: {image.shape}")
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            self.logger.info(f"Processed image shape: {processed_image.shape}")
            
            # Validate model
            if self.model is None:
                self.logger.error("Model not loaded")
                return None
            
            # Get embedding
            embedding = self.model.predict(processed_image, verbose=0)
            self.logger.info(f"Raw embedding shape: {embedding.shape}")
            
            # Validate embedding shape
            if embedding.shape[-1] != 128:
                self.logger.error(f"Unexpected embedding dimension: {embedding.shape}, expected 128")
                return None
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            flattened = embedding.flatten()
            self.logger.info(f"Final embedding shape: {flattened.shape}")
            
            return flattened
            
        except Exception as e:
            self.logger.error(f"Failed to extract face embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Validate embedding shapes
            if embedding1.shape != embedding2.shape:
                self.logger.error(f"Embedding shape mismatch: {embedding1.shape} vs {embedding2.shape}")
                return 0.0
            
            # Ensure embeddings are 1D arrays
            if embedding1.ndim > 1:
                embedding1 = embedding1.flatten()
            if embedding2.ndim > 1:
                embedding2 = embedding2.flatten()
            
            # Check if embeddings are valid (not empty or all zeros)
            if np.all(embedding1 == 0) or np.all(embedding2 == 0):
                self.logger.warning("One or both embeddings are zero vectors")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                self.logger.warning("One or both embeddings have zero norm")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clamp similarity to valid range [-1, 1]
            similarity = np.clip(similarity, -1.0, 1.0)
            
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
            
            # Validate embedding before storing
            if embedding.shape != (128,) and embedding.shape != (1, 128):
                self.logger.error(f"Invalid embedding shape for user {user_id}: {embedding.shape}")
                return {
                    "success": False,
                    "error": f"Invalid embedding shape: {embedding.shape}"
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
            # Validate and convert input image
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Image is not a numpy array, got type: {type(image)}")
                if isinstance(image, list):
                    self.logger.error(f"Image is a list with length: {len(image)}")
                    # Try to convert list to numpy array
                    try:
                        image = np.array(image)
                        self.logger.info(f"Converted list to numpy array with shape: {image.shape}")
                    except Exception as conv_error:
                        self.logger.error(f"Failed to convert list to numpy array: {conv_error}")
                        return None
                else:
                    return None
            
            if image.size == 0:
                self.logger.error("Image is empty")
                return None
                
            self.logger.info(f"Input image type: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")
            
            # Extract face embedding
            query_embedding = self._extract_face_embedding(image)
            
            if query_embedding is None:
                return None
            
            # Find best match
            best_match = None
            best_similarity = 0.0
            
            for user_id, stored_embedding in self.face_embeddings.items():
                # Validate stored embedding shape
                if stored_embedding.shape != query_embedding.shape:
                    self.logger.warning(f"Skipping user {user_id} due to embedding shape mismatch: {stored_embedding.shape} vs {query_embedding.shape}")
                    continue
                
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
        """Save face database to Supabase Storage"""
        try:
            data = {
                'embeddings': self.face_embeddings,
                'database': self.face_database
            }
            
            # Save directly to Supabase Storage
            if self.supabase_client:
                db_data = pickle.dumps(data)
                
                # Try to upload with upsert option first
                try:
                    self.supabase_client.storage.from_("autosos").upload(
                        self.database_path,
                        db_data,
                        {"content-type": "application/octet-stream", "upsert": "true"}
                    )
                    self.logger.info(f"Face database saved to Supabase Storage: {self.database_path}")
                except Exception as upload_error:
                    # If upsert fails, try delete then upload
                    if "409" in str(upload_error) or "Duplicate" in str(upload_error):
                        self.logger.info("File exists, deleting and re-uploading...")
                        try:
                            # Delete existing file
                            self.supabase_client.storage.from_("autosos").remove([self.database_path])
                            # Upload new file
                            self.supabase_client.storage.from_("autosos").upload(
                                self.database_path,
                                db_data,
                                {"content-type": "application/octet-stream"}
                            )
                            self.logger.info(f"Face database updated in Supabase Storage: {self.database_path}")
                        except Exception as delete_error:
                            self.logger.error(f"Failed to delete and re-upload: {delete_error}")
                            raise
                    else:
                        raise upload_error
            else:
                self.logger.warning("Supabase client not available, database not saved")
                
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
    
    def is_face_registered(self, user_id: str) -> bool:
        """Check if a user has a registered face"""
        try:
            # Check local database first
            if user_id in self.face_database:
                return True
            
            # Check Supabase database if available
            if self.supabase_client:
                try:
                    result = self.supabase_client.rpc('get_face_embedding', {
                        'p_user_id': user_id
                    }).execute()
                    
                    if result.data and len(result.data) > 0:
                        return True
                except Exception as e:
                    self.logger.warning(f"Failed to check Supabase for user {user_id}: {e}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking face registration for user {user_id}: {e}")
            return False
    
    def remove_face(self, user_id: str) -> Dict[str, Any]:
        """Remove a registered face"""
        try:
            # Remove from local database
            if user_id in self.face_embeddings:
                del self.face_embeddings[user_id]
            
            if user_id in self.face_database:
                del self.face_database[user_id]
            
            # Remove from Supabase database if available
            if self.supabase_client:
                try:
                    # Deactivate face embedding in Supabase
                    result = self.supabase_client.rpc('deactivate_face_embedding', {
                        'p_user_id': user_id
                    }).execute()
                    
                    if result.data:
                        self.logger.info(f"Face deactivated in Supabase for user {user_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to deactivate face in Supabase for user {user_id}: {e}")
            
            # Save updated database
            self._save_database()
            
            self.logger.info(f"Face removed for user {user_id}")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "Face removed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to remove face for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }