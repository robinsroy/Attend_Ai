#!/usr/bin/env python3
"""
Debug script to test auth route step by step
"""

from app import create_app, db
from app.models import User
from app.services.auth_service import AuthService
from app.utils.validators import validate_login_data
from werkzeug.security import check_password_hash
import json

def test_full_auth_flow():
    """Test the complete authentication flow"""
    app = create_app()
    
    with app.app_context():
        print("🔍 Testing complete authentication flow...")
        
        # Step 1: Check user exists
        print("\n1. Checking if user exists in database...")
        user = User.query.filter_by(username='teacher1').first()
        if user:
            print(f"✅ User found: {user.username} ({user.full_name})")
            print(f"   Email: {user.email}")
            print(f"   Is active: {user.is_active}")
        else:
            print("❌ User not found in database")
            return
        
        # Step 2: Test password hash
        print("\n2. Testing password hash...")
        is_password_valid = check_password_hash(user.password_hash, 'password123')
        print(f"   Password valid: {is_password_valid}")
        
        # Step 3: Test validation
        print("\n3. Testing request validation...")
        test_data = {'username': 'teacher1', 'password': 'password123'}
        is_valid, errors = validate_login_data(test_data)
        print(f"   Validation passed: {is_valid}")
        if not is_valid:
            print(f"   Errors: {errors}")
        
        # Step 4: Test AuthService
        print("\n4. Testing AuthService...")
        auth_user = AuthService.authenticate_user('teacher1', 'password123')
        if auth_user:
            print(f"✅ AuthService successful: {auth_user.username}")
        else:
            print("❌ AuthService failed")
        
        # Step 5: Simulate the actual route logic
        print("\n5. Simulating route logic...")
        
        # Validate input data
        is_valid, errors = validate_login_data(test_data)
        if not is_valid:
            print(f"❌ Validation failed: {errors}")
            return
        
        # Authenticate user
        user = AuthService.authenticate_user(test_data['username'], test_data['password'])
        if not user:
            print("❌ Authentication failed in route simulation")
            return
        
        if not user.is_active:
            print("❌ User account is not active")
            return
        
        print("✅ Route simulation successful!")
        print(f"   User: {user.username}")
        print(f"   Ready to generate tokens")

if __name__ == "__main__":
    test_full_auth_flow()
