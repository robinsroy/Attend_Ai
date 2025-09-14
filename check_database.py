#!/usr/bin/env python3
"""
Check database records for enrollment frames
"""
import sys
import os
sys.path.append('backend')

def check_enrollment_frames():
    """Check enrollment frames in database"""
    try:
        from backend.app import create_app, db
        from backend.app.models.enrollment_frame import EnrollmentFrame
        from backend.app.models.student import Student
        
        app = create_app()
        with app.app_context():
            # Check enrollment frames
            frames = EnrollmentFrame.query.all()
            print(f"📊 Total enrollment frames in database: {len(frames)}")
            
            if frames:
                print("\n🎬 Enrollment Frames:")
                for frame in frames[-10:]:  # Show last 10 frames
                    print(f"  - {frame.student_name} ({frame.roll_number}) Frame #{frame.frame_number}")
                    print(f"    File: {frame.file_path}")
                    print(f"    Size: {frame.file_size} bytes")
                    print(f"    Time: {frame.timestamp}")
                    print()
            
            # Check students
            students = Student.query.all()
            print(f"👥 Total students in database: {len(students)}")
            
            if students:
                print("\n👤 Students:")
                for student in students:
                    enrollment_status = "✅ Enrolled" if getattr(student, 'is_enrolled', False) else "⏳ Pending"
                    print(f"  - {student.full_name} ({student.roll_number}) - {enrollment_status}")
            
            # Check file storage
            storage_base = os.path.join(os.getcwd(), 'backend', 'storage', 'enrollment_frames')
            if os.path.exists(storage_base):
                student_dirs = [d for d in os.listdir(storage_base) if os.path.isdir(os.path.join(storage_base, d))]
                print(f"\n📁 Student directories in storage: {len(student_dirs)}")
                
                for student_dir in student_dirs:
                    dir_path = os.path.join(storage_base, student_dir)
                    frame_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
                    print(f"  - {student_dir}: {len(frame_files)} frame files")
            else:
                print(f"\n❌ Storage directory not found: {storage_base}")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        print("\nTo fix this, run:")
        print("cd D:/Attend_Ai/backend")
        print("python -c \"from app import create_app, db; app = create_app(); app.app_context().push(); db.create_all(); print('Database initialized!')\"")

if __name__ == "__main__":
    print("🔍 Checking Enrollment Database Records...\n")
    check_enrollment_frames()
