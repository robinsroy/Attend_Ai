#!/usr/bin/env python3
"""
Test script to verify API endpoints are working
"""

import requests
import json

def test_login():
    """Test the login endpoint"""
    url = "http://localhost:5000/api/auth/login"
    data = {
        "username": "teacher1",
        "password": "password123"
    }
    
    try:
        print(f"Testing login at: {url}")
        print(f"Data: {data}")
        
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Login successful!")
            return response.json()
        else:
            print("❌ Login failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error testing login: {e}")
        return None

def test_health():
    """Test the health endpoint"""
    url = "http://localhost:5000/api/health"
    
    try:
        print(f"Testing health at: {url}")
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Health check successful!")
            return True
        else:
            print("❌ Health check failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing health: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing API endpoints...")
    print("\n" + "="*50)
    
    # Test health first
    print("1. Testing Health Endpoint")
    test_health()
    
    print("\n" + "="*50)
    
    # Test login
    print("2. Testing Login Endpoint")
    test_login()
    
    print("\n" + "="*50)
    print("✅ API testing complete!")
