-- Create database and user
CREATE DATABASE attend_ai;
CREATE USER attend_ai_user WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE attend_ai TO attend_ai_user;

-- Switch to attend_ai database
\c attend_ai;

-- Create tables in correct order (respecting foreign key dependencies)

-- 1. Users table (no dependencies)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(20) DEFAULT 'teacher',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Classes table (no dependencies)
CREATE TABLE classes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    section VARCHAR(10) NOT NULL,
    grade VARCHAR(10) NOT NULL,
    subject VARCHAR(50) NOT NULL,
    academic_year VARCHAR(10) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Periods table (depends on classes)
CREATE TABLE periods (
    id SERIAL PRIMARY KEY,
    class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
    period_number INTEGER NOT NULL,
    period_name VARCHAR(50) NOT NULL,
    start_time VARCHAR(10) NOT NULL,
    end_time VARCHAR(10) NOT NULL,
    days_of_week VARCHAR(20) NOT NULL,
    is_active BOOLEAN DEFAULT true
);

-- 4. Students table (depends on classes)
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    roll_number VARCHAR(20) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
    master_embedding BYTEA,
    enrollment_video_path VARCHAR(255),
    face_quality_score FLOAT,
    guardian_name VARCHAR(100),
    guardian_phone VARCHAR(15),
    date_of_birth DATE,
    address TEXT,
    is_active BOOLEAN DEFAULT true,
    enrollment_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Attendance Sessions table (depends on classes, periods, users)
CREATE TABLE attendance_sessions (
    id SERIAL PRIMARY KEY,
    class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
    period_id INTEGER REFERENCES periods(id) ON DELETE CASCADE,
    teacher_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_date DATE DEFAULT CURRENT_DATE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    total_students INTEGER DEFAULT 0,
    present_students INTEGER DEFAULT 0,
    absent_students INTEGER DEFAULT 0,
    camera_source VARCHAR(100),
    detection_threshold FLOAT DEFAULT 0.6,
    session_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Attendance Records table (depends on sessions, students, users)
CREATE TABLE attendance_records (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES attendance_sessions(id) ON DELETE CASCADE,
    student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'present',
    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score FLOAT,
    detection_method VARCHAR(20) DEFAULT 'auto',
    face_image_path VARCHAR(255),
    coordinates VARCHAR(100),
    manual_override BOOLEAN DEFAULT false,
    override_reason VARCHAR(255),
    override_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_students_class_id ON students(class_id);
CREATE INDEX idx_students_roll_number ON students(roll_number);
CREATE INDEX idx_attendance_sessions_date ON attendance_sessions(session_date);
CREATE INDEX idx_attendance_sessions_class ON attendance_sessions(class_id);
CREATE INDEX idx_attendance_records_session ON attendance_records(session_id);
CREATE INDEX idx_attendance_records_student ON attendance_records(student_id);
CREATE INDEX idx_periods_class_id ON periods(class_id);

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO attend_ai_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO attend_ai_user;
