# Attendance Management System API

## Authentication Routes
- `POST /auth/login` - Teacher login
- `POST /auth/logout` - Teacher logout
- `GET /auth/profile` - Get current user profile

## Student Management Routes
- `POST /students/enroll` - Enroll new student
- `GET /students` - Get all students
- `GET /students/{id}` - Get specific student
- `PUT /students/{id}` - Update student information
- `DELETE /students/{id}` - Remove student

## Class Management Routes
- `GET /classes` - Get all classes
- `POST /classes` - Create new class
- `GET /classes/{id}/students` - Get students in class
- `GET /classes/{id}/periods` - Get periods for class

## Attendance Routes
- `POST /attendance/session/start` - Start attendance session
- `POST /attendance/session/end` - End attendance session
- `POST /attendance/mark` - Mark student attendance
- `GET /attendance/session/{id}` - Get session details
- `GET /attendance/reports` - Generate attendance reports

## Face Recognition Routes
- `POST /face/detect` - Detect faces in image/video
- `POST /face/recognize` - Recognize face against database
- `POST /face/embedding` - Generate face embedding
- `GET /face/quality` - Check face image quality
