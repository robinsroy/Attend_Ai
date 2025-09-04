#!/usr/bin/env python3
"""
Create dummy data for testing the Attend.AI system
"""

from app import create_app, db
from app.models import User, Class, Student, Period
from datetime import time

def create_dummy_data():
    app = create_app()
    with app.app_context():
        print('🚀 Creating dummy data for testing...')
        
        # Get the teacher
        teacher = User.query.first()
        if not teacher:
            print('❌ No teacher found')
            return
        
        print(f'Found teacher: {teacher.username}')
        
        # Create dummy classes
        classes_data = [
            ('10-A Computer Science', 'A', '10', 'Computer Science'),
            ('10-B Mathematics', 'B', '10', 'Mathematics'),
            ('11-A Physics', 'A', '11', 'Physics')
        ]
        
        for name, section, grade, subject in classes_data:
            existing_class = Class.query.filter_by(name=name).first()
            if not existing_class:
                new_class = Class(
                    name=name,
                    section=section,
                    grade=grade,
                    subject=subject,
                    academic_year='2024-25',
                    teacher_id=teacher.id,
                    room_number=f'Room-{section}01'
                )
                db.session.add(new_class)
                print(f'✅ Created class: {name}')
            else:
                print(f'Class already exists: {name}')
        
        db.session.commit()
        
        # Create periods for each class
        all_classes = Class.query.all()
        periods_data = [
            ('Morning Assembly', time(8, 30), time(8, 45)),
            ('1st Period', time(9, 0), time(9, 45)),
            ('2nd Period', time(9, 45), time(10, 30)),
            ('Break', time(10, 30), time(10, 45)),
            ('3rd Period', time(10, 45), time(11, 30)),
            ('4th Period', time(11, 30), time(12, 15)),
            ('Lunch', time(12, 15), time(13, 0)),
            ('5th Period', time(13, 0), time(13, 45)),
            ('6th Period', time(13, 45), time(14, 30))
        ]
        
        for class_obj in all_classes:
            print(f'Creating periods for {class_obj.name}...')
            for name, start, end in periods_data:
                existing_period = Period.query.filter_by(name=name, class_id=class_obj.id).first()
                if not existing_period:
                    period = Period(
                        name=name,
                        start_time=start,
                        end_time=end,
                        class_id=class_obj.id,
                        day_of_week='Monday-Friday'
                    )
                    db.session.add(period)
            
        db.session.commit()
        
        # Create dummy students
        students_data = [
            ('John Doe', 'CS001', 'Mr. Doe', '1234567890'),
            ('Jane Smith', 'CS002', 'Mrs. Smith', '0987654321'),
            ('Bob Johnson', 'CS003', 'Mr. Johnson', '1122334455'),
            ('Alice Brown', 'CS004', 'Mrs. Brown', '5544332211'),
            ('Charlie Wilson', 'CS005', 'Mr. Wilson', '9988776655')
        ]
        
        first_class = Class.query.first()
        if first_class:
            print(f'Creating students for {first_class.name}...')
            for name, roll, guardian, phone in students_data:
                existing_student = Student.query.filter_by(roll_number=roll).first()
                if not existing_student:
                    student = Student(
                        full_name=name,
                        roll_number=roll,
                        class_id=first_class.id,
                        guardian_name=guardian,
                        guardian_phone=phone,
                        enrollment_status='completed',
                        address='Sample Address'
                    )
                    db.session.add(student)
                    print(f'✅ Created student: {name} ({roll})')
                else:
                    print(f'Student already exists: {name}')
        
        db.session.commit()
        
        # Show summary
        classes_count = Class.query.count()
        students_count = Student.query.count()
        periods_count = Period.query.count()
        
        print(f'''
🎉 Database setup complete!
📚 Classes: {classes_count}
👥 Students: {students_count}
⏰ Periods: {periods_count}
        ''')

if __name__ == '__main__':
    create_dummy_data()
