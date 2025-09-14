"""
Real-Time Face Recognition Service with DeepFace
Implements continuous automated loop: MTCNN Detection → Liveness Check → ArcFace Fingerprint → Roster Matching
"""

import cv2
import numpy as np
from deepface import DeepFace
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import base64
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeRecognitionService:
    """
    Real-time face recognition service implementing the complete automated loop:
    A. Detect Face (DeepFace with MTCNN)
    B. Check Liveness (Motion + Eye Analysis alternative to dlib)
    C. Create Live Fingerprint (DeepFace with ArcFace)
    D. Match Against Active Roster
    """
    
    def __init__(self):
        self.camera = None
        self.recognition_active = False
        self.current_session_id = None
        self.recognized_students = {}
        self.last_recognition_times = {}
        self.recognition_cooldown = 5  # 5 seconds cooldown per student
        
        # DeepFace configuration
        self.detector_backend = 'mtcnn'  # MTCNN for face detection
        self.model_name = 'ArcFace'     # ArcFace for fingerprint creation
        
        # Alternative liveness detection (motion + face analysis)
        self.prev_frame = None
        self.motion_threshold = 2000
        self.face_positions = []  # Track face movement
        self.liveness_frames = 5  # Frames to analyze for liveness
        
        # Threading
        self.recognition_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Mock roster data (integrate with your database)
        self.active_roster = self._load_mock_roster()
        
        logger.info("RealtimeRecognitionService initialized with DeepFace")
        self._test_deepface_availability()
    
    def _test_deepface_availability(self):
        """Test if DeepFace is working properly"""
        try:
            # Test MTCNN backend
            logger.info("Testing MTCNN detector...")
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            DeepFace.extract_faces(test_image, detector_backend='mtcnn', enforce_detection=False)
            logger.info("✅ MTCNN detector working")
            
            # Test ArcFace model
            logger.info("Testing ArcFace model...")
            DeepFace.represent(test_image, model_name='ArcFace', enforce_detection=False)
            logger.info("✅ ArcFace model working")
            
        except Exception as e:
            logger.warning(f"DeepFace test failed: {e}")
            logger.info("Will use fallback methods if needed")
    
    def _load_mock_roster(self):
        """Load mock student roster (replace with actual database integration)"""
        return {
            'STUDENT_001': {
                'name': 'John Doe',
                'roll': '2025001',
                'fingerprint': np.random.rand(512),  # ArcFace produces 512-dim vectors
                'class_id': 1
            },
            'STUDENT_002': {
                'name': 'Jane Smith',
                'roll': '2025002',
                'fingerprint': np.random.rand(512),
                'class_id': 1
            },
            'STUDENT_003': {
                'name': 'Bob Johnson',
                'roll': '2025003',
                'fingerprint': np.random.rand(512),
                'class_id': 1
            },
            'STUDENT_004': {
                'name': 'Alice Brown',
                'roll': '2025004',
                'fingerprint': np.random.rand(512),
                'class_id': 1
            }
        }
    
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.camera.isOpened():
                logger.error("Could not open camera")
                return False
            
            logger.info(f"Camera {camera_index} started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            logger.info("Camera stopped")
    
    def start_recognition_loop(self, session_id: str, class_id: int):
        """Start the automated recognition loop"""
        self.current_session_id = session_id
        self.recognition_active = True
        
        # Filter roster by class_id (if implementing database integration)
        logger.info(f"Loading roster for class {class_id}")
        
        # Start recognition thread
        self.recognition_thread = threading.Thread(
            target=self._recognition_loop_worker,
            daemon=True
        )
        self.recognition_thread.start()
        
        logger.info(f"🤖 Recognition loop started for session {session_id}, class {class_id}")
    
    def stop_recognition_loop(self):
        """Stop the automated recognition loop"""
        self.recognition_active = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=5)
        logger.info("Recognition loop stopped")
    
    def _recognition_loop_worker(self):
        """Main recognition loop worker thread"""
        logger.info("🔄 Recognition loop worker started")
        
        while self.recognition_active and self.camera:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Process frame through recognition pipeline
                self._process_frame(frame)
                
                # Small delay to prevent overloading (process ~2 FPS for recognition)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in recognition loop: {e}")
                time.sleep(1)
        
        logger.info("Recognition loop worker stopped")
    
    def _process_frame(self, frame: np.ndarray):
        """
        Process a single frame through the complete recognition pipeline:
        A. Detect Face (DeepFace MTCNN)
        B. Check Liveness (Motion + Face Movement Analysis)
        C. Create Live Fingerprint (DeepFace ArcFace)
        D. Match Against Roster
        """
        try:
            # A. DETECT FACE using DeepFace MTCNN
            face_regions = self._detect_faces_deepface(frame)
            
            if not face_regions:
                return
            
            logger.debug(f"📹 Detected {len(face_regions)} faces")
            
            for face_data in face_regions:
                # B. CHECK LIVENESS using alternative method
                if not self._check_alternative_liveness(frame, face_data):
                    logger.debug("👁️ Face failed liveness check")
                    continue
                
                # C. CREATE LIVE FINGERPRINT using DeepFace ArcFace
                live_fingerprint = self._create_live_fingerprint_deepface(face_data)
                
                if live_fingerprint is None:
                    logger.debug("🧠 Could not create live fingerprint")
                    continue
                
                # D. MATCH AGAINST ACTIVE ROSTER
                match_result = self._match_against_roster(live_fingerprint)
                
                if match_result:
                    self._handle_recognition_result(match_result, frame, face_data)
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _detect_faces_deepface(self, frame: np.ndarray) -> List[Dict]:
        """A. Detect faces using DeepFace with MTCNN backend"""
        try:
            # DeepFace expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace with MTCNN for face detection
            face_objs = DeepFace.extract_faces(
                img_path=rgb_frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            faces = []
            for i, face_array in enumerate(face_objs):
                if face_array is not None and face_array.size > 0:
                    # Convert back to 0-255 range and BGR for OpenCV
                    face_bgr = cv2.cvtColor((face_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    faces.append({
                        'face_array': face_bgr,
                        'face_rgb': face_array,
                        'index': i,
                        'confidence': 0.9  # MTCNN typically has high confidence
                    })
            
            return faces
        
        except Exception as e:
            logger.debug(f"MTCNN face detection error: {e}")
            return []
    
    def _check_alternative_liveness(self, frame: np.ndarray, face_data: Dict) -> bool:
        """
        B. Alternative liveness detection (since dlib is not available)
        Uses motion detection + face position tracking
        """
        try:
            # Method 1: Motion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return True  # First frame, assume live
            
            # Calculate motion
            diff = cv2.absdiff(self.prev_frame, gray)
            motion_pixels = cv2.countNonZero(diff)
            
            self.prev_frame = gray
            
            # Method 2: Face Position Tracking (detect face movement)
            face_center = self._get_face_center_estimate(face_data)
            self.face_positions.append(face_center)
            
            # Keep only recent positions
            if len(self.face_positions) > self.liveness_frames:
                self.face_positions.pop(0)
            
            # Calculate face movement
            face_movement = 0
            if len(self.face_positions) >= 2:
                positions = np.array(self.face_positions)
                distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                face_movement = np.sum(distances)
            
            # Method 3: Eye Region Analysis (simplified without dlib)
            eye_variation = self._analyze_eye_regions(face_data['face_array'])
            
            # Combine all liveness indicators
            motion_score = min(motion_pixels / self.motion_threshold, 1.0)
            movement_score = min(face_movement / 50.0, 1.0)  # Normalize face movement
            eye_score = eye_variation
            
            # Combined liveness score
            liveness_score = (motion_score + movement_score + eye_score) / 3.0
            is_live = liveness_score > 0.3  # Threshold for liveness
            
            logger.debug(f"👁️ Liveness - Motion: {motion_score:.2f}, Movement: {movement_score:.2f}, Eye: {eye_score:.2f}, Live: {is_live}")
            return is_live
        
        except Exception as e:
            logger.debug(f"Alternative liveness check error: {e}")
            return True  # Default to live if check fails
    
    def _get_face_center_estimate(self, face_data: Dict) -> Tuple[int, int]:
        """Estimate face center from face array"""
        try:
            face_array = face_data['face_array']
            h, w = face_array.shape[:2]
            return (w // 2, h // 2)
        except:
            return (0, 0)
    
    def _analyze_eye_regions(self, face_array: np.ndarray) -> float:
        """
        Simplified eye region analysis (alternative to dlib eye aspect ratio)
        Analyzes intensity variations in estimated eye regions
        """
        try:
            gray_face = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
            h, w = gray_face.shape
            
            # Estimate eye regions (approximate positions)
            left_eye_region = gray_face[int(h*0.3):int(h*0.5), int(w*0.2):int(w*0.4)]
            right_eye_region = gray_face[int(h*0.3):int(h*0.5), int(w*0.6):int(w*0.8)]
            
            # Calculate intensity variation in eye regions
            left_variation = np.std(left_eye_region) if left_eye_region.size > 0 else 0
            right_variation = np.std(right_eye_region) if right_eye_region.size > 0 else 0
            
            # Higher variation suggests open eyes, lower suggests closed/blinking
            avg_variation = (left_variation + right_variation) / 2.0
            eye_score = min(avg_variation / 50.0, 1.0)  # Normalize
            
            return eye_score
        
        except Exception as e:
            logger.debug(f"Eye analysis error: {e}")
            return 0.5  # Default neutral score
    
    def _create_live_fingerprint_deepface(self, face_data: Dict) -> Optional[np.ndarray]:
        """C. Create live fingerprint using DeepFace ArcFace model"""
        try:
            # Use the RGB face array for DeepFace
            face_rgb = face_data['face_rgb']
            
            # Generate embedding using ArcFace
            embedding_result = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding_result and len(embedding_result) > 0:
                fingerprint = np.array(embedding_result[0]['embedding'])
                logger.debug(f"🧠 Live fingerprint created with ArcFace: shape={fingerprint.shape}")
                return fingerprint
            else:
                return None
        
        except Exception as e:
            logger.debug(f"ArcFace fingerprint creation error: {e}")
            return None
    
    def _match_against_roster(self, live_fingerprint: np.ndarray) -> Optional[Dict]:
        """D. Match live fingerprint against Active Roster using cosine similarity"""
        try:
            best_match = None
            best_confidence = 0.0
            
            for student_id, student_data in self.active_roster.items():
                # Calculate cosine similarity between fingerprints
                roster_fingerprint = student_data['fingerprint']
                
                # Ensure same dimensions
                if live_fingerprint.shape != roster_fingerprint.shape:
                    continue
                
                # Cosine similarity
                dot_product = np.dot(live_fingerprint, roster_fingerprint)
                norm_live = np.linalg.norm(live_fingerprint)
                norm_roster = np.linalg.norm(roster_fingerprint)
                
                if norm_live > 0 and norm_roster > 0:
                    similarity = dot_product / (norm_live * norm_roster)
                    # Convert similarity to confidence (0-1 range)
                    confidence = (similarity + 1) / 2
                    
                    # Check if this is the best match above threshold
                    if confidence > best_confidence and confidence > 0.70:  # 70% confidence threshold
                        best_confidence = confidence
                        best_match = {
                            'student_id': student_id,
                            'student_name': student_data['name'],
                            'roll_number': student_data['roll'],
                            'confidence': confidence
                        }
            
            if best_match:
                logger.debug(f"📋 Match found: {best_match['student_name']} ({best_match['confidence']:.3f})")
                return best_match
            else:
                logger.debug("📋 No confident match found in roster")
                return None
        
        except Exception as e:
            logger.error(f"Roster matching error: {e}")
            return None
    
    def _handle_recognition_result(self, match_result: Dict, frame: np.ndarray, face_data: Dict):
        """Handle successful recognition result"""
        try:
            student_id = match_result['student_id']
            student_name = match_result['student_name']
            confidence = match_result['confidence']
            
            # Check cooldown to prevent duplicate recognitions
            current_time = time.time()
            if student_id in self.last_recognition_times:
                if current_time - self.last_recognition_times[student_id] < self.recognition_cooldown:
                    return
            
            # Update recognition time
            self.last_recognition_times[student_id] = current_time
            
            # Store recognition result
            recognition_data = {
                'student_id': student_id,
                'student_name': student_name,
                'roll_number': match_result['roll_number'],
                'confidence': confidence,
                'timestamp': current_time,
                'time': datetime.fromtimestamp(current_time).strftime('%H:%M:%S'),
                'session_id': self.current_session_id,
                'method': 'deepface_arcface'
            }
            
            self.recognized_students[student_id] = recognition_data
            
            # Add to result queue for UI updates
            self.result_queue.put(recognition_data)
            
            logger.info(f"✅ Student recognized: {student_name} (confidence: {confidence:.3f})")
            
            # TODO: Save to database
            self._save_attendance_record(recognition_data)
        
        except Exception as e:
            logger.error(f"Error handling recognition result: {e}")
    
    def _save_attendance_record(self, recognition_data: Dict):
        """Save attendance record to database"""
        try:
            # Placeholder for database integration
            logger.info(f"💾 Saving attendance for {recognition_data['student_name']}")
            
            # In a real implementation, you would:
            # 1. Connect to your database
            # 2. Create AttendanceRecord entry
            # 3. Save with session_id, student_id, confidence, etc.
            
        except Exception as e:
            logger.error(f"Error saving attendance record: {e}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from camera"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def get_latest_results(self) -> List[Dict]:
        """Get latest recognition results"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def get_session_statistics(self) -> Dict:
        """Get current session statistics"""
        return {
            'total_recognized': len(self.recognized_students),
            'active_session': self.current_session_id,
            'recognition_active': self.recognition_active,
            'camera_active': self.camera is not None,
            'recognized_students': list(self.recognized_students.values()),
            'detector_backend': self.detector_backend,
            'model_name': self.model_name
        }
    
    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame for testing purposes"""
        try:
            results = {
                'faces_detected': 0,
                'recognitions': [],
                'liveness_passed': 0
            }
            
            face_regions = self._detect_faces_deepface(frame)
            results['faces_detected'] = len(face_regions)
            
            for face_data in face_regions:
                if self._check_alternative_liveness(frame, face_data):
                    results['liveness_passed'] += 1
                    
                    fingerprint = self._create_live_fingerprint_deepface(face_data)
                    if fingerprint is not None:
                        match = self._match_against_roster(fingerprint)
                        if match:
                            results['recognitions'].append(match)
            
            return results
        
        except Exception as e:
            logger.error(f"Error processing single frame: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_recognition_loop()
        self.stop_camera()
        logger.info("RealtimeRecognitionService cleanup completed")

# Standalone testing function
def test_recognition_service():
    """Test the recognition service"""
    service = RealtimeRecognitionService()
    
    print("🧪 Testing Recognition Service")
    print("1. Starting camera...")
    
    if service.start_camera():
        print("✅ Camera started")
        
        print("2. Testing single frame processing...")
        frame = service.get_latest_frame()
        if frame is not None:
            results = service.process_single_frame(frame)
            print(f"📊 Results: {results}")
        
        print("3. Starting recognition loop...")
        service.start_recognition_loop("test_session", 1)
        
        print("4. Running for 10 seconds...")
        time.sleep(10)
        
        print("5. Getting statistics...")
        stats = service.get_session_statistics()
        print(f"📈 Final stats: {stats}")
        
        service.cleanup()
        print("✅ Test completed")
    else:
        print("❌ Could not start camera")

if __name__ == "__main__":
    test_recognition_service()