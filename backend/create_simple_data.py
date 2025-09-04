"""
Simple sample data creation using direct SQL
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash
from datetime import datetime

def create_simple_sample_data():
    """Create sample data using direct SQL to avoid model dependencies"""
    
    # Connect to the SQLite database
    db_path = os.path.join(os.path.dirname(__file__), '..', 'instance', 'attendance.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🚀 Creating sample data...")
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(200) NOT NULL,
            role VARCHAR(50) DEFAULT 'teacher',
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            teacher_id INTEGER,
            academic_year VARCHAR(20),
            section VARCHAR(10),
            room_number VARCHAR(20),
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name VARCHAR(200) NOT NULL,
            roll_number VARCHAR(50) UNIQUE NOT NULL,
            class_id INTEGER NOT NULL,
            guardian_name VARCHAR(200),
            guardian_phone VARCHAR(20),
            address TEXT,
            enrollment_status VARCHAR(20) DEFAULT 'completed',
            face_embedding TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS periods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            class_id INTEGER NOT NULL,
            subject VARCHAR(100),
            start_time TIME NOT NULL,
            end_time TIME NOT NULL,
            day_of_week VARCHAR(20),
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        )
    ''')
    
    # Insert users
    print("👤 Creating users...")
    users_data = [
        ('admin', 'admin@attend.ai', generate_password_hash('admin123'), 'System Administrator', 'admin'),
        ('teacher1', 'john.doe@school.edu', generate_password_hash('teacher123'), 'John Doe', 'teacher'),
        ('teacher2', 'jane.smith@school.edu', generate_password_hash('teacher123'), 'Jane Smith', 'teacher')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO users (username, email, password_hash, full_name, role)
        VALUES (?, ?, ?, ?, ?)
    ''', users_data)
    
    # Insert classes
    print("🏫 Creating classes...")
    classes_data = [
        ('Class 10A', 2, '2024-25', 'A', '101'),
        ('Class 10B', 3, '2024-25', 'B', '102'),
        ('Class 9A', 2, '2024-25', 'A', '201')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO classes (name, teacher_id, academic_year, section, room_number)
        VALUES (?, ?, ?, ?, ?)
    ''', classes_data)
    
    # Insert periods
    print("📅 Creating periods...")
    periods_data = [
        ('Mathematics - Period 1', 1, 'Mathematics', '08:00', '08:45', 'Monday'),
        ('Mathematics - Period 2', 1, 'Mathematics', '09:00', '09:45', 'Tuesday'),
        ('Science - Period 1', 2, 'Science', '10:00', '10:45', 'Monday'),
        ('English - Period 1', 3, 'English', '11:00', '11:45', 'Wednesday')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO periods (name, class_id, subject, start_time, end_time, day_of_week)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', periods_data)
    
    # Insert students
    print("👥 Creating students...")
    students_data = [
        # Class 10A students
        ('Aarav Sharma', '10A001', 1, 'Raj Sharma', '+91-9876543210', '123 Main Street, City'),
        ('Priya Patel', '10A002', 1, 'Amit Patel', '+91-9876543211', '456 Park Avenue, City'),
        ('Rohan Kumar', '10A003', 1, 'Suresh Kumar', '+91-9876543212', '789 Oak Street, City'),
        ('Ananya Singh', '10A004', 1, 'Vikram Singh', '+91-9876543213', '321 Pine Street, City'),
        ('Arjun Reddy', '10A005', 1, 'Krishna Reddy', '+91-9876543214', '654 Elm Street, City'),
        
        # Class 10B students
        ('Kavya Nair', '10B001', 2, 'Ravi Nair', '+91-9876543215', '987 Cedar Street, City'),
        ('Vikash Gupta', '10B002', 2, 'Manoj Gupta', '+91-9876543216', '147 Maple Street, City'),
        ('Shreya Joshi', '10B003', 2, 'Ramesh Joshi', '+91-9876543217', '258 Birch Street, City'),
        
        # Class 9A students
        ('Karthik Iyer', '9A001', 3, 'Sanjay Iyer', '+91-9876543218', '369 Willow Street, City'),
        ('Meera Agarwal', '9A002', 3, 'Deepak Agarwal', '+91-9876543219', '741 Poplar Street, City')
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO students (full_name, roll_number, class_id, guardian_name, guardian_phone, address)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', students_data)
    
    # Commit changes
    conn.commit()
    
    # Get counts
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM classes')
    class_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM students')
    student_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM periods')
    period_count = cursor.fetchone()[0]
    
    conn.close()
    
    print("✅ Sample data created successfully!")
    print(f"\n📊 Summary:")
    print(f"   👤 Users: {user_count}")
    print(f"   🏫 Classes: {class_count}")
    print(f"   📅 Periods: {period_count}")
    print(f"   👥 Students: {student_count}")
    
    print(f"\n🔑 Login Credentials:")
    print(f"   Admin: username='admin', password='admin123'")
    print(f"   Teacher 1: username='teacher1', password='teacher123'")
    print(f"   Teacher 2: username='teacher2', password='teacher123'")
    
    print(f"\n🏫 Classes Created:")
    print(f"   - Class 10A (Teacher: John Doe)")
    print(f"   - Class 10B (Teacher: Jane Smith)")
    print(f"   - Class 9A (Teacher: John Doe)")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Open http://localhost:8501 (Frontend)")
    print(f"   2. Login with any teacher credentials")
    print(f"   3. Test enrollment and attendance features")
    print(f"   4. Upload student photos for face recognition")

if __name__ == '__main__':
    create_simple_sample_data()
