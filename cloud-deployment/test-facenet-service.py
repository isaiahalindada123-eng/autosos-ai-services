#!/usr/bin/env python3
"""
Test script for FaceNet service
"""

import requests
import json
import base64
import io
from PIL import Image
import numpy as np

# Service URL
FACENET_URL = "https://autosos-ai-services-1.onrender.com"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{FACENET_URL}/health")
        print(f"Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    try:
        response = requests.get(f"{FACENET_URL}/metrics")
        print(f"Metrics: {response.status_code}")
        if response.status_code == 200:
            print("Metrics endpoint working")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Metrics check failed: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a simple 100x100 RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_face_count():
    """Test face count endpoint"""
    try:
        response = requests.get(f"{FACENET_URL}/face-count")
        print(f"Face Count: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Face count check failed: {e}")
        return False

def test_recognize_base64():
    """Test face recognition with base64 image"""
    try:
        test_image = create_test_image()
        
        payload = {
            "image_base64": test_image,
            "user_id": "test_user"
        }
        
        response = requests.post(
            f"{FACENET_URL}/recognize-base64",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Recognize Base64: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Recognize base64 check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing FaceNet Service...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Metrics", test_metrics),
        ("Face Count", test_face_count),
        ("Recognize Base64", test_recognize_base64),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()
