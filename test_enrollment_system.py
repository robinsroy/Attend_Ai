#!/usr/bin/env python3
"""
Test script to verify the enrollment system is working properly
"""
import requests
import json
import os

def test_backend_running():
    """Test if backend is running"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print("❌ Backend health check failed")
            return False
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def test_database_tables():
    """Test if database tables exist"""
    try:
        # Test frame saving endpoint (this will fail if table doesn't exist)
        test_data = {
            "student_name": "Test Student",
            "roll_number": "TEST001",
            "frame_number": 1,
            "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
        
        response = requests.post(
            'http://localhost:5000/api/face/save_frame',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(test_data)
        )
        
        if response.status_code == 200:
            print("✅ Database tables exist and frame saving works")
            return True
        else:
            print(f"❌ Frame saving failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_storage_directory():
    """Test if storage directory exists"""
    storage_path = "D:/Attend_Ai/backend/storage/enrollment_frames"
    if os.path.exists(storage_path):
        print("✅ Storage directory exists")
        return True
    else:
        print(f"❌ Storage directory missing: {storage_path}")
        print("   Run: mkdir -p D:/Attend_Ai/backend/storage/enrollment_frames")
        return False

def test_frontend_running():
    """Test if frontend is running"""
    try:
        response = requests.get('http://localhost:8501')
        if response.status_code == 200:
            print("✅ Frontend is running")
            return True
        else:
            print("❌ Frontend not accessible")
            return False
    except Exception as e:
        print(f"❌ Frontend not accessible: {e}")
        return False

def main():
    print("🔍 Testing Attend.AI Enrollment System...\n")
    
    tests = [
        ("Backend API", test_backend_running),
        ("Database Tables", test_database_tables),
        ("Storage Directory", test_storage_directory),
        ("Frontend App", test_frontend_running)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    print("=" * 50)
    if all(results):
        print("🎉 ALL TESTS PASSED! System is ready for enrollment.")
        print("\n📋 Next Steps:")
        print("1. Go to http://localhost:8501")
        print("2. Fill student information")
        print("3. Start video recording")
        print("4. Check console for frame saving messages")
        print("5. Verify frames are saved to database")
    else:
        print("❌ Some tests failed. Please fix the issues above before testing enrollment.")
        
        print("\n🔧 Quick Fix Commands:")
        print("# Create database tables:")
        print("cd D:/Attend_Ai/backend")
        print("python -c \"from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('Tables created!')\"")
        print()
        print("# Create storage directory:")
        print("mkdir -p D:/Attend_Ai/backend/storage/enrollment_frames")
        print()
        print("# Start backend (if not running):")
        print("cd D:/Attend_Ai/backend && python run.py")
        print()
        print("# Start frontend (if not running):")
        print("cd D:/Attend_Ai/frontend && streamlit run app.py")

if __name__ == "__main__":
    main()
