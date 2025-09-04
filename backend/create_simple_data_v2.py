#!/usr/bin/env python3
"""
Simple script to create basic sample data for testing
"""

from app import create_app, db
from app.models import User, Student, Class, Period
from werkzeug.security import generate_password_hash
from datetime import datetime, time

def create_simple_data():
    """Create basic sample data for testing"""
    app = create_app()
    
    with app.app_context():
        # Clear existing data
        db.drop_all()
        db.create_all()
        
        print("🚀 Creating simple sample data...")
        
        # Create teacher user
        print("👤 Creating teacher...")
        teacher = User(
            username='teacher1',
            email='teacher1@school.edu',
            password_hash=generate_password_hash('password123'),
            full_name='John Teacher',
            role='teacher'
        )
        db.session.add(teacher)
        db.session.commit()
        
        # Create class
        print("📚 Creating class...")
        class1 = Class(
            name='10-A Computer Science',
            section='A',
            grade='10',
            subject='Computer Science',
            academic_year='2024-25',
            teacher_id=teacher.id
        )
        db.session.add(class1)
        db.session.commit()
        
        # Create period
        print("⏰ Creating period...")
        period1 = Period(
            name='1st Period',
            start_time=time(9, 0),
            end_time=time(9, 45),
            class_id=class1.id
        )
        db.session.add(period1)
        db.session.commit()
        
        # Create students
        print("👥 Creating students...")
        students_data = [
            ('CS001', 'Alice Johnson'),
            ('CS002', 'Bob Smith'),
            ('CS003', 'Carol Davis')
        ]
        
        for roll_no, name in students_data:
            student = Student(
                roll_number=roll_no,
                full_name=name,
                class_id=class1.id,
                enrollment_status='completed'
            )
            db.session.add(student)
        
        db.session.commit()
        
        print("✅ Sample data created successfully!")
        print(f"   - Teacher: {teacher.username} (password: password123)")
        print(f"   - Class: {class1.name}")
        print(f"   - Students: {len(students_data)}")

if __name__ == '__main__':
    create_simple_data()
