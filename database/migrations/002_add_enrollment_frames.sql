-- Migration: Add enrollment_frames table for storing video frames during student enrollment
-- Date: 2025-09-09

CREATE TABLE IF NOT EXISTS enrollment_frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name VARCHAR(255) NOT NULL,
    roll_number VARCHAR(50) NOT NULL,
    frame_number INTEGER NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER,
    timestamp DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_student_roll ON enrollment_frames(student_name, roll_number);
CREATE INDEX IF NOT EXISTS idx_frame_number ON enrollment_frames(frame_number);
CREATE INDEX IF NOT EXISTS idx_timestamp ON enrollment_frames(timestamp);

-- Add comment
INSERT OR IGNORE INTO schema_version (version, description, applied_at) 
VALUES (2, 'Add enrollment_frames table for video frame storage', datetime('now'));
