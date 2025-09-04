-- Sample data for testing the Attend.AI system

-- Insert sample users (teachers)
INSERT INTO users (username, email, password_hash, full_name, role) VALUES
('teacher1', 'teacher1@school.edu', '$2b$12$sample_hash_1', 'John Smith', 'teacher'),
('teacher2', 'teacher2@school.edu', '$2b$12$sample_hash_2', 'Sarah Johnson', 'teacher'),
('admin1', 'admin@school.edu', '$2b$12$sample_hash_3', 'Michael Brown', 'admin');

-- Insert sample classes
INSERT INTO classes (name, section, grade, subject, academic_year) VALUES
('10-A Physics', 'A', '10', 'Physics', '2024-25'),
('10-B Computer Science', 'B', '10', 'Computer Science', '2024-25'),
('11-A Mathematics', 'A', '11', 'Mathematics', '2024-25'),
('11-B Chemistry', 'B', '11', 'Chemistry', '2024-25');

-- Insert sample periods for classes
INSERT INTO periods (class_id, period_number, period_name, start_time, end_time, days_of_week) VALUES
-- For 10-A Physics (class_id = 1)
(1, 1, 'Period 1', '09:00', '10:00', 'Mon,Wed,Fri'),
(1, 2, 'Period 2', '10:15', '11:15', 'Tue,Thu'),
-- For 10-B Computer Science (class_id = 2)
(2, 1, 'Period 1', '11:30', '12:30', 'Mon,Wed,Fri'),
(2, 2, 'Period 2', '13:30', '14:30', 'Tue,Thu'),
-- For 11-A Mathematics (class_id = 3)
(3, 1, 'Period 1', '09:00', '10:00', 'Tue,Thu'),
(3, 2, 'Period 2', '14:45', '15:45', 'Mon,Wed,Fri'),
-- For 11-B Chemistry (class_id = 4)
(4, 1, 'Period 1', '10:15', '11:15', 'Mon,Wed,Fri'),
(4, 2, 'Period 2', '11:30', '12:30', 'Tue,Thu');

-- Insert sample students
INSERT INTO students (roll_number, full_name, class_id, guardian_name, guardian_phone, enrollment_status) VALUES
-- 10-A Physics students
('10A001', 'Alice Anderson', 1, 'Robert Anderson', '+1234567890', 'completed'),
('10A002', 'Bob Baker', 1, 'Linda Baker', '+1234567891', 'completed'),
('10A003', 'Charlie Chen', 1, 'David Chen', '+1234567892', 'completed'),
('10A004', 'Diana Davis', 1, 'Mary Davis', '+1234567893', 'pending'),
('10A005', 'Edward Evans', 1, 'James Evans', '+1234567894', 'completed'),

-- 10-B Computer Science students
('10B001', 'Frank Foster', 2, 'Susan Foster', '+1234567895', 'completed'),
('10B002', 'Grace Green', 2, 'Paul Green', '+1234567896', 'completed'),
('10B003', 'Henry Harris', 2, 'Carol Harris', '+1234567897', 'completed'),
('10B004', 'Ivy Jackson', 2, 'Mark Jackson', '+1234567898', 'pending'),
('10B005', 'Jack Johnson', 2, 'Lisa Johnson', '+1234567899', 'completed'),

-- 11-A Mathematics students
('11A001', 'Kelly King', 3, 'Tom King', '+1234567800', 'completed'),
('11A002', 'Leo Lopez', 3, 'Anna Lopez', '+1234567801', 'completed'),
('11A003', 'Mia Martinez', 3, 'Carlos Martinez', '+1234567802', 'completed'),

-- 11-B Chemistry students
('11B001', 'Noah Nelson', 4, 'Helen Nelson', '+1234567803', 'completed'),
('11B002', 'Olivia O\'Connor', 4, 'Sean O\'Connor', '+1234567804', 'completed'),
('11B003', 'Peter Parker', 4, 'May Parker', '+1234567805', 'pending');

-- Insert sample attendance session
INSERT INTO attendance_sessions (class_id, period_id, teacher_id, session_date, start_time, status, total_students) VALUES
(1, 1, 1, CURRENT_DATE, CURRENT_TIMESTAMP, 'active', 5);

-- Insert sample attendance records
INSERT INTO attendance_records (session_id, student_id, status, confidence_score, detection_method) VALUES
(1, 1, 'present', 0.95, 'auto'),
(1, 2, 'present', 0.88, 'auto'),
(1, 3, 'present', 0.92, 'auto');

-- Update session statistics
UPDATE attendance_sessions SET present_students = 3, absent_students = 2 WHERE id = 1;
