"""
Sample data initialization script for Attend.AI
Creates test users, classes, students, and periods for demonstration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models.user import User
from app.models.student import Student
from app.models.class_model import Class
from app.models.period import Period
from werkzeug.security import generate_password_hash
from datetime import datetime, time

def create_sample_data():
    """Create sample data for testing"""
    
    app = create_app()
    
    with app.app_context():
        print("🚀 Creating sample data...")
        
        # Create test users (teachers)
        print("👤 Creating users...")
        
        # Admin user
        admin_user = User(
            username='admin',
            email='admin@attend.ai',
            password_hash=generate_password_hash('admin123'),
            full_name='System Administrator',
            role='admin'
        )
        
        # Teacher users
        teacher1 = User(
            username='teacher1',
            email='john.doe@school.edu',
            password_hash=generate_password_hash('teacher123'),
            full_name='John Doe',
            role='teacher'
        )
        
        teacher2 = User(
            username='teacher2',
            email='jane.smith@school.edu',
            password_hash=generate_password_hash('teacher123'),
            full_name='Jane Smith',
            role='teacher'
        )
        
        db.session.add_all([admin_user, teacher1, teacher2])
        db.session.commit()
        
        # Create classes
        print("🏫 Creating classes...")
        
        class_10a = Class(
            name='Class 10A',
            teacher_id=teacher1.id,
            academic_year='2024-25',
            section='A',
            room_number='101'
        )
        
        class_10b = Class(
            name='Class 10B',
            teacher_id=teacher2.id,
            academic_year='2024-25',
            section='B',
            room_number='102'
        )
        
        class_9a = Class(
            name='Class 9A',
            teacher_id=teacher1.id,
            academic_year='2024-25',
            section='A',
            room_number='201'
        )
        
        db.session.add_all([class_10a, class_10b, class_9a])
        db.session.commit()
        
        # Create periods
        print("📅 Creating periods...")
        
        # Math periods for Class 10A
        math_period1 = Period(
            name='Mathematics - Period 1',
            class_id=class_10a.id,
            subject='Mathematics',
            start_time=time(8, 0),  # 8:00 AM
            end_time=time(8, 45),   # 8:45 AM
            day_of_week='Monday',
            is_active=True
        )
        
        math_period2 = Period(
            name='Mathematics - Period 2',
            class_id=class_10a.id,
            subject='Mathematics',
            start_time=time(9, 0),  # 9:00 AM
            end_time=time(9, 45),   # 9:45 AM
            day_of_week='Tuesday',
            is_active=True
        )
        
        # Science periods for Class 10B
        science_period1 = Period(
            name='Science - Period 1',
            class_id=class_10b.id,
            subject='Science',
            start_time=time(10, 0),  # 10:00 AM
            end_time=time(10, 45),   # 10:45 AM
            day_of_week='Monday',
            is_active=True
        )
        
        # English periods for Class 9A
        english_period1 = Period(
            name='English - Period 1',
            class_id=class_9a.id,
            subject='English',
            start_time=time(11, 0),  # 11:00 AM
            end_time=time(11, 45),   # 11:45 AM
            day_of_week='Wednesday',
            is_active=True
        )
        
        db.session.add_all([math_period1, math_period2, science_period1, english_period1])
        db.session.commit()
        
        # Create sample students
        print("👥 Creating students...")
        
        # Students for Class 10A
        students_10a = [
            {
                'full_name': 'Aarav Sharma',
                'roll_number': '10A001',
                'guardian_name': 'Raj Sharma',
                'guardian_phone': '+91-9876543210',
                'address': '123 Main Street, City'
            },
            {
                'full_name': 'Priya Patel',
                'roll_number': '10A002',
                'guardian_name': 'Amit Patel',
                'guardian_phone': '+91-9876543211',
                'address': '456 Park Avenue, City'
            },
            {
                'full_name': 'Rohan Kumar',
                'roll_number': '10A003',
                'guardian_name': 'Suresh Kumar',
                'guardian_phone': '+91-9876543212',
                'address': '789 Oak Street, City'
            },
            {
                'full_name': 'Ananya Singh',
                'roll_number': '10A004',
                'guardian_name': 'Vikram Singh',
                'guardian_phone': '+91-9876543213',
                'address': '321 Pine Street, City'
            },
            {
                'full_name': 'Arjun Reddy',
                'roll_number': '10A005',
                'guardian_name': 'Krishna Reddy',
                'guardian_phone': '+91-9876543214',
                'address': '654 Elm Street, City'
            }
        ]
        
        # Students for Class 10B
        students_10b = [
            {
                'full_name': 'Kavya Nair',
                'roll_number': '10B001',
                'guardian_name': 'Ravi Nair',
                'guardian_phone': '+91-9876543215',
                'address': '987 Cedar Street, City'
            },
            {
                'full_name': 'Vikash Gupta',
                'roll_number': '10B002',
                'guardian_name': 'Manoj Gupta',
                'guardian_phone': '+91-9876543216',
                'address': '147 Maple Street, City'
            },
            {
                'full_name': 'Shreya Joshi',
                'roll_number': '10B003',
                'guardian_name': 'Ramesh Joshi',
                'guardian_phone': '+91-9876543217',
                'address': '258 Birch Street, City'
            }
        ]
        
        # Students for Class 9A
        students_9a = [
            {
                'full_name': 'Karthik Iyer',
                'roll_number': '9A001',
                'guardian_name': 'Sanjay Iyer',
                'guardian_phone': '+91-9876543218',
                'address': '369 Willow Street, City'
            },
            {
                'full_name': 'Meera Agarwal',
                'roll_number': '9A002',
                'guardian_name': 'Deepak Agarwal',
                'guardian_phone': '+91-9876543219',
                'address': '741 Poplar Street, City'
            }
        ]
        
        # Create student records
        all_students = []
        
        for student_data in students_10a:
            student = Student(
                full_name=student_data['full_name'],
                roll_number=student_data['roll_number'],
                class_id=class_10a.id,
                guardian_name=student_data['guardian_name'],
                guardian_phone=student_data['guardian_phone'],
                address=student_data['address'],
                enrollment_status='completed'
            )
            all_students.append(student)
        
        for student_data in students_10b:
            student = Student(
                full_name=student_data['full_name'],
                roll_number=student_data['roll_number'],
                class_id=class_10b.id,
                guardian_name=student_data['guardian_name'],
                guardian_phone=student_data['guardian_phone'],
                address=student_data['address'],
                enrollment_status='completed'
            )
            all_students.append(student)
        
        for student_data in students_9a:
            student = Student(
                full_name=student_data['full_name'],
                roll_number=student_data['roll_number'],
                class_id=class_9a.id,
                guardian_name=student_data['guardian_name'],
                guardian_phone=student_data['guardian_phone'],
                address=student_data['address'],
                enrollment_status='completed'
            )
            all_students.append(student)
        
        db.session.add_all(all_students)
        db.session.commit()
        
        print("✅ Sample data created successfully!")
        print("\n📊 Summary:")
        print(f"   👤 Users: {User.query.count()}")
        print(f"   🏫 Classes: {Class.query.count()}")
        print(f"   📅 Periods: {Period.query.count()}")
        print(f"   👥 Students: {Student.query.count()}")
        
        print("\n🔑 Login Credentials:")
        print("   Admin: username='admin', password='admin123'")
        print("   Teacher 1: username='teacher1', password='teacher123'")
        print("   Teacher 2: username='teacher2', password='teacher123'")
        
        print("\n🏫 Classes Created:")
        print("   - Class 10A (Teacher: John Doe)")
        print("   - Class 10B (Teacher: Jane Smith)")
        print("   - Class 9A (Teacher: John Doe)")
        
        print("\n🎯 Next Steps:")
        print("   1. Open http://localhost:8501 (Frontend)")
        print("   2. Login with any teacher credentials")
        print("   3. Test enrollment and attendance features")
        print("   4. Upload student photos for face recognition")

if __name__ == '__main__':
    create_sample_data()
