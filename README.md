# Attend.AI - Intelligent Face Recognition Attendance System

## Project Overview
A comprehensive facial recognition-based attendance system designed for educational institutions. The system uses ArcFace embeddings for high-accuracy face recognition and provides real-time attendance marking capabilities.

## Architecture

### Backend (`/backend`)
- **FastAPI/Flask** - RESTful API server
- **SQLAlchemy** - Database ORM
- **JWT Authentication** - Secure user sessions
- **Redis** - Caching for embeddings and sessions

### Face Recognition (`/face_recognition`)
- **ArcFace Model** - 512-dimensional embeddings
- **OpenCV** - Video processing and face detection
- **MTCNN** - Face detection and alignment
- **NumPy** - Mathematical operations for embeddings

### Frontend (`/frontend`)
- **React.js** - User interface
- **WebRTC** - Camera access and video streaming
- **Material-UI/Tailwind** - Component library
- **Axios** - API communication

### Database (`/database`)
- **PostgreSQL/MySQL** - Primary database
- **Migrations** - Database schema management
- **Seeders** - Initial data setup

## Project Structure

```
Attend_Ai/
├── backend/                    # Backend API server
│   ├── app/
│   │   ├── models/            # Database models
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Business logic
│   │   ├── controllers/       # Request handlers
│   │   └── utils/             # Helper functions
│   ├── config/                # Configuration files
│   └── requirements.txt       # Python dependencies
│
├── face_recognition/          # Face recognition module
│   ├── models/                # AI model files
│   ├── processors/            # Image/video processing
│   ├── embeddings/            # Embedding generation
│   └── matching/              # Face matching algorithms
│
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Main application pages
│   │   └── services/          # API service calls
│   └── package.json           # Node.js dependencies
│
├── database/                  # Database management
│   ├── migrations/            # Schema migrations
│   └── seeders/               # Sample data
│
├── storage/                   # File storage
│   ├── embeddings/            # Stored face embeddings
│   ├── videos/                # Enrollment videos
│   └── logs/                  # Application logs
│
├── tests/                     # Test files
├── docs/                      # Documentation
└── docker-compose.yml         # Container orchestration
```

## Key Features

### Phase 1: Student Registration
- Teacher login and authentication
- Student information entry (name, roll number, class)
- Guided video capture for facial profile
- ArcFace embedding generation and storage
- Class assignment management

### Phase 2: Daily Attendance
- Class and period selection
- Real-time face detection and recognition
- Automatic attendance marking
- Live visual feedback
- Session management and reporting

## Getting Started

1. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **Database Setup**
   ```bash
   cd database
   # Run migrations
   ```

## Technology Stack

- **Backend**: Python, FastAPI/Flask, SQLAlchemy
- **Frontend**: React.js, WebRTC, Material-UI
- **AI/ML**: ArcFace, OpenCV, MTCNN, NumPy
- **Database**: PostgreSQL/MySQL, Redis
- **Deployment**: Docker, Docker Compose

## Development Workflow

1. Database models and schema design
2. Backend API development
3. Face recognition module implementation
4. Frontend component development
5. Integration and testing
6. Deployment and optimization
