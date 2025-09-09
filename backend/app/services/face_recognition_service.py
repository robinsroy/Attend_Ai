"""
Face Recognition Service using DeepFace with ArcFace embeddings
Handles video capture, frame extraction, filtering, and embedding generation
"""
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import os
import time
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Advanced face recognition service for student enrollment and attendance"""
    
    def __init__(self):
        """Initialize the face recognition service"""
        self.model_name = "ArcFace"
        self.detector_backend = "mtcnn"
        self.embedding_size = 512
        
        # Quality thresholds
        self.min_face_confidence = 0.95
        self.min_blur_threshold = 100  # Laplacian variance threshold
        self.min_brightness = 50
        self.max_brightness = 200
        
    def capture_enrollment_video(self, duration_seconds: int = 20) -> Optional[str]:
        """
        Capture a 20-second enrollment video with guided prompts
        Returns the path to the captured video file
        """
        try:
            # Create temporary video file
            video_path = os.path.join(tempfile.gettempdir(), f"enrollment_{int(time.time())}.avi")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Cannot access camera")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 30
            frame_size = (1280, 720)
            out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
            
            start_time = time.time()
            frame_count = 0
            
            logger.info(f"Starting {duration_seconds}-second enrollment video capture")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                elapsed_time = time.time() - start_time
                remaining_time = duration_seconds - elapsed_time
                
                if elapsed_time >= duration_seconds:
                    break
                
                # Add guidance text based on time
                frame_with_guidance = self._add_guidance_overlay(frame, elapsed_time, duration_seconds)
                
                # Write frame to video
                out.write(frame_with_guidance)
                frame_count += 1
            
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Video capture completed. {frame_count} frames captured")
            return video_path
            
        except Exception as e:
            logger.error(f"Error capturing enrollment video: {str(e)}")
            return None
    
    def _add_guidance_overlay(self, frame, elapsed_time: float, total_duration: int):
        """Add guidance text overlay to video frame"""
        frame_copy = frame.copy()
        height, width = frame_copy.shape[:2]
        
        # Determine guidance message based on elapsed time
        progress = elapsed_time / total_duration
        
        if progress < 0.3:
            message = "Look straight at the camera"
            color = (0, 255, 0)  # Green
        elif progress < 0.6:
            message = "Slowly turn your head to the left"
            color = (0, 165, 255)  # Orange
        elif progress < 0.9:
            message = "Slowly turn your head to the right"
            color = (0, 165, 255)  # Orange
        else:
            message = "Look straight - Almost done!"
            color = (0, 255, 0)  # Green
        
        # Add semi-transparent overlay
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
        
        # Add guidance text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_copy, message, (50, 35), font, 1, color, 2)
        
        # Add time remaining
        remaining = total_duration - elapsed_time
        time_text = f"Time remaining: {remaining:.1f}s"
        cv2.putText(frame_copy, time_text, (50, 65), font, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def extract_and_filter_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video and filter for quality
        Returns list of high-quality frames suitable for embedding generation
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return []
            
            good_frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames (analyze every 5th frame for efficiency)
                if frame_count % 5 != 0:
                    continue
                
                # Quality checks
                if self._is_frame_good_quality(frame):
                    good_frames.append(frame)
                    
                    # Limit to max 50 good frames
                    if len(good_frames) >= 50:
                        break
            
            cap.release()
            
            logger.info(f"Extracted {len(good_frames)} good quality frames from {frame_count} total frames")
            return good_frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def _is_frame_good_quality(self, frame: np.ndarray) -> bool:
        """
        Check if frame meets quality criteria for face recognition
        """
        try:
            # Convert to grayscale for quality checks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Check brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < self.min_brightness or mean_brightness > self.max_brightness:
                return False
            
            # Check blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.min_blur_threshold:
                return False
            
            # Check if face is detected with high confidence
            try:
                faces = DeepFace.extract_faces(
                    frame, 
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
                
                if len(faces) != 1:  # Should have exactly one face
                    return False
                
                # Additional face quality checks could be added here
                return True
                
            except Exception:
                return False
            
        except Exception as e:
            logger.error(f"Error checking frame quality: {str(e)}")
            return False
    
    def generate_embeddings_from_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate ArcFace embeddings from good quality frames
        """
        embeddings = []
        
        for i, frame in enumerate(frames):
            try:
                # Generate embedding using DeepFace with ArcFace
                embedding = DeepFace.represent(
                    frame,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
                
                if embedding and len(embedding) > 0:
                    # DeepFace returns list of embeddings, take the first one
                    emb_vector = np.array(embedding[0]["embedding"])
                    embeddings.append(emb_vector)
                    
            except Exception as e:
                logger.warning(f"Failed to generate embedding for frame {i}: {str(e)}")
                continue
        
        logger.info(f"Generated {len(embeddings)} embeddings from {len(frames)} frames")
        return embeddings
    
    def create_master_embedding(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create master embedding by averaging all individual embeddings
        """
        if not embeddings:
            logger.error("No embeddings provided for master embedding creation")
            return None
        
        try:
            # Convert to numpy array and calculate mean
            embeddings_array = np.array(embeddings)
            master_embedding = np.mean(embeddings_array, axis=0)
            
            # Normalize the master embedding
            master_embedding = master_embedding / np.linalg.norm(master_embedding)
            
            logger.info(f"Created master embedding from {len(embeddings)} individual embeddings")
            return master_embedding
            
        except Exception as e:
            logger.error(f"Error creating master embedding: {str(e)}")
            return None
    
    def process_enrollment_video(self, video_path: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Complete enrollment processing pipeline
        Returns master embedding and quality score
        """
        try:
            # Extract and filter frames
            good_frames = self.extract_and_filter_frames(video_path)
            
            if len(good_frames) < 5:  # Need minimum 5 good frames
                logger.error(f"Insufficient good quality frames: {len(good_frames)}")
                return None, 0.0
            
            # Generate embeddings
            embeddings = self.generate_embeddings_from_frames(good_frames)
            
            if len(embeddings) < 3:  # Need minimum 3 embeddings
                logger.error(f"Insufficient embeddings generated: {len(embeddings)}")
                return None, 0.0
            
            # Create master embedding
            master_embedding = self.create_master_embedding(embeddings)
            
            if master_embedding is None:
                return None, 0.0
            
            # Calculate quality score based on consistency of embeddings
            quality_score = self._calculate_quality_score(embeddings)
            
            # Cleanup video file
            try:
                os.remove(video_path)
            except:
                pass
            
            return master_embedding, quality_score
            
        except Exception as e:
            logger.error(f"Error processing enrollment video: {str(e)}")
            return None, 0.0
    
    def _calculate_quality_score(self, embeddings: List[np.ndarray]) -> float:
        """
        Calculate quality score based on consistency of embeddings
        """
        if len(embeddings) < 2:
            return 0.0
        
        try:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Quality score is the mean similarity
            quality_score = np.mean(similarities)
            
            # Normalize to 0-100 scale
            quality_score = max(0, min(100, quality_score * 100))
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def recognize_face_live(self, frame: np.ndarray, master_embeddings: List[Tuple[int, np.ndarray]], threshold: float = 0.6) -> Optional[int]:
        """
        Recognize face in live frame against master embeddings
        Returns student_id if recognized, None if not
        """
        try:
            # Generate embedding for current frame
            embedding = DeepFace.represent(
                frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            if not embedding or len(embedding) == 0:
                return None
            
            current_embedding = np.array(embedding[0]["embedding"])
            current_embedding = current_embedding / np.linalg.norm(current_embedding)
            
            # Compare against all master embeddings
            best_match_id = None
            best_similarity = -1
            
            for student_id, master_embedding in master_embeddings:
                similarity = np.dot(current_embedding, master_embedding)
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match_id = student_id
            
            return best_match_id
            
        except Exception as e:
            logger.error(f"Error recognizing face: {str(e)}")
            return None
    
    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy embedding to bytes for database storage"""
        return embedding.tobytes()
    
    def deserialize_embedding(self, embedding_bytes: bytes) -> np.ndarray:
        """Convert bytes from database back to numpy array"""
        return np.frombuffer(embedding_bytes, dtype=np.float64).reshape(-1)
    
    # Legacy methods for backward compatibility
    @staticmethod
    def detect_faces(image_data: bytes) -> Dict[str, Any]:
        """Detect faces in image data - legacy method for compatibility"""
        try:
            # Convert bytes to cv2 image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'faces': [], 'processing_time': 0, 'error': 'Invalid image data'}
            
            # Load OpenCV face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            start_time = cv2.getTickCount()
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            end_time = cv2.getTickCount()
            
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            
            # Format face data
            face_data = []
            for (x, y, w, h) in faces:
                face_data.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': 0.8  # OpenCV doesn't provide confidence, so we use a default
                })
            
            return {
                'faces': face_data,
                'processing_time': processing_time,
                'image_shape': image.shape
            }
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return {'faces': [], 'processing_time': 0, 'error': str(e)}
    
    @staticmethod
    def generate_embedding(image_data: bytes) -> Dict[str, Any]:
        """Generate face embedding from image data"""
        try:
            # First detect faces
            detection_result = FaceRecognitionService.detect_faces(image_data)
            
            if not detection_result['faces']:
                return {
                    'success': False,
                    'message': 'No face detected in image',
                    'embedding': None
                }
            
            if len(detection_result['faces']) > 1:
                return {
                    'success': False,
                    'message': 'Multiple faces detected, please ensure only one face is visible',
                    'embedding': None
                }
            
            # For now, generate a mock embedding (512-dimensional)
            # In production, this would use ArcFace or similar model
            mock_embedding = np.random.rand(512).astype(np.float32)
            
            face = detection_result['faces'][0]
            quality_score = FaceRecognitionService._calculate_quality_score(image_data, face)
            
            return {
                'success': True,
                'embedding': mock_embedding,
                'quality_score': quality_score,
                'face_coordinates': face,
                'message': 'Embedding generated successfully'
            }
            
        except Exception as e:
            logger.error(f"Generate embedding error: {str(e)}")
            return {
                'success': False,
                'message': f'Embedding generation failed: {str(e)}',
                'embedding': None
            }
    
    @staticmethod
    def recognize_face(image_data: bytes, class_id: int, threshold: float = 0.6) -> Dict[str, Any]:
        """Recognize face against class database"""
        try:
            # Generate embedding for input image
            embedding_result = FaceRecognitionService.generate_embedding(image_data)
            
            if not embedding_result['success']:
                return {
                    'student_found': False,
                    'message': embedding_result['message']
                }
            
            input_embedding = embedding_result['embedding']
            
            # This would normally compare against stored embeddings
            # For now, return mock result
            mock_confidence = 0.85
            
            if mock_confidence >= threshold:
                return {
                    'student_found': True,
                    'student': {
                        'id': 1,
                        'name': 'Mock Student',
                        'roll_number': 'MOCK001'
                    },
                    'confidence_score': mock_confidence,
                    'face_coordinates': embedding_result['face_coordinates']
                }
            else:
                return {
                    'student_found': False,
                    'max_confidence': mock_confidence,
                    'message': 'No matching student found above threshold'
                }
                
        except Exception as e:
            logger.error(f"Face recognition error: {str(e)}")
            return {
                'student_found': False,
                'message': f'Recognition failed: {str(e)}'
            }
    
    @staticmethod
    def check_face_quality(image_data: bytes) -> Dict[str, Any]:
        """Check face image quality for enrollment"""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    'quality_score': 0.0,
                    'is_good_quality': False,
                    'issues': ['Invalid image data'],
                    'recommendations': ['Please provide a valid image']
                }
            
            # Detect faces first
            detection_result = FaceRecognitionService.detect_faces(image_data)
            
            if not detection_result['faces']:
                return {
                    'quality_score': 0.0,
                    'is_good_quality': False,
                    'issues': ['No face detected'],
                    'recommendations': ['Ensure face is clearly visible']
                }
            
            if len(detection_result['faces']) > 1:
                return {
                    'quality_score': 0.3,
                    'is_good_quality': False,
                    'issues': ['Multiple faces detected'],
                    'recommendations': ['Ensure only one face is visible']
                }
            
            face = detection_result['faces'][0]
            
            # Calculate quality metrics
            quality_score = FaceRecognitionService._calculate_quality_score(image_data, face)
            
            issues = []
            recommendations = []
            
            # Check face size
            face_area = face['width'] * face['height']
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            if face_ratio < 0.05:
                issues.append('Face too small')
                recommendations.append('Move closer to camera')
            elif face_ratio > 0.8:
                issues.append('Face too large')
                recommendations.append('Move away from camera')
            
            # Check image brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness < 80:
                issues.append('Image too dark')
                recommendations.append('Improve lighting')
            elif brightness > 200:
                issues.append('Image too bright')
                recommendations.append('Reduce lighting')
            
            # Check blur
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 100:
                issues.append('Image blurry')
                recommendations.append('Hold camera steady')
            
            is_good_quality = quality_score >= 0.7 and len(issues) == 0
            
            return {
                'quality_score': quality_score,
                'is_good_quality': is_good_quality,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Face quality check error: {str(e)}")
            return {
                'quality_score': 0.0,
                'is_good_quality': False,
                'issues': [f'Quality check failed: {str(e)}'],
                'recommendations': ['Please try again']
            }
    
    @staticmethod
    def check_liveness(image_data: bytes) -> Dict[str, Any]:
        """Basic liveness detection"""
        try:
            # This is a very basic implementation
            # In production, you would use specialized anti-spoofing models
            
            detection_result = FaceRecognitionService.detect_faces(image_data)
            
            if not detection_result['faces']:
                return {
                    'is_live': False,
                    'confidence': 0.0,
                    'anti_spoofing_score': 0.0
                }
            
            # Mock liveness detection
            # Real implementation would analyze texture, movement, etc.
            mock_liveness_score = 0.85
            
            return {
                'is_live': mock_liveness_score > 0.5,
                'confidence': mock_liveness_score,
                'anti_spoofing_score': mock_liveness_score
            }
            
        except Exception as e:
            logger.error(f"Liveness check error: {str(e)}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'anti_spoofing_score': 0.0
            }
    
    @staticmethod
    def process_enrollment_video(video_path: str) -> Dict[str, Any]:
        """Process enrollment video to generate master embedding"""
        try:
            if not os.path.exists(video_path):
                return {
                    'success': False,
                    'message': 'Video file not found'
                }
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {
                    'success': False,
                    'message': 'Unable to open video file'
                }
            
            embeddings = []
            frames_processed = 0
            
            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Generate embedding for frame
                embedding_result = FaceRecognitionService.generate_embedding(frame_bytes)
                
                if embedding_result['success'] and embedding_result['quality_score'] > 0.7:
                    embeddings.append(embedding_result['embedding'])
                    frames_processed += 1
                
                # Process every 5th frame to save time
                for _ in range(4):
                    ret, _ = cap.read()
                    if not ret:
                        break
            
            cap.release()
            
            if len(embeddings) < 3:
                return {
                    'success': False,
                    'message': 'Insufficient good quality frames found',
                    'frames_processed': frames_processed
                }
            
            # Average the embeddings to create master embedding
            master_embedding = np.mean(embeddings, axis=0)
            
            # Calculate quality score
            quality_score = min(0.9, 0.5 + (len(embeddings) * 0.1))
            
            return {
                'success': True,
                'master_embedding': master_embedding.tobytes(),  # Convert to bytes for storage
                'quality_score': quality_score,
                'frames_processed': frames_processed,
                'message': 'Master embedding created successfully'
            }
            
        except Exception as e:
            logger.error(f"Process enrollment video error: {str(e)}")
            return {
                'success': False,
                'message': f'Video processing failed: {str(e)}'
            }
    
    @staticmethod
    def _calculate_quality_score(image_data: bytes, face_coords: Dict[str, int]) -> float:
        """Calculate face quality score"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract face region
            x, y, w, h = face_coords['x'], face_coords['y'], face_coords['width'], face_coords['height']
            face_region = image[y:y+h, x:x+w]
            
            # Calculate various quality metrics
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            blur_quality = min(1.0, blur_score / 500.0)  # Normalize
            
            # Brightness check
            brightness = np.mean(gray_face)
            brightness_quality = 1.0 - abs(brightness - 128) / 128.0
            
            # Face size quality
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            size_ratio = face_area / image_area
            size_quality = min(1.0, size_ratio * 10) if size_ratio < 0.1 else min(1.0, (1 - size_ratio) * 2)
            
            # Combine scores
            overall_quality = (blur_quality * 0.4 + brightness_quality * 0.3 + size_quality * 0.3)
            
            return round(overall_quality, 2)
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {str(e)}")
            return 0.5  # Default medium quality

    @staticmethod
    def process_video_enrollment(student_id: int, video_data: bytes) -> Dict[str, Any]:
        """
        Process video for student enrollment and generate master embedding
        This method replaces the previous video processing workflow
        """
        try:
            from app.models.student import Student
            from app import db
            
            # Get student record
            student = Student.query.get(student_id)
            if not student:
                return {
                    'success': False,
                    'error': 'Student not found'
                }
            
            # Save video data to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_data)
            temp_file.close()
            
            try:
                # Extract frames from video
                frames_data = FaceRecognitionService.extract_frames_from_video(temp_file.name)
                
                if not frames_data or len(frames_data) == 0:
                    return {
                        'success': False,
                        'error': 'No frames could be extracted from video'
                    }
                
                # Filter high-quality frames
                high_quality_frames = []
                quality_scores = []
                
                for frame_data in frames_data:
                    # Detect faces in frame
                    face_result = FaceRecognitionService.detect_faces(frame_data['data'])
                    
                    if face_result['faces']:
                        # Use first detected face
                        face = face_result['faces'][0]
                        
                        # Calculate quality score
                        quality_score = FaceRecognitionService._calculate_quality_score(
                            frame_data['data'], 
                            face['coordinates']
                        )
                        
                        if quality_score > 0.6:  # Only use good quality frames
                            high_quality_frames.append(frame_data['data'])
                            quality_scores.append(quality_score)
                
                if not high_quality_frames:
                    return {
                        'success': False,
                        'error': 'No high-quality face frames found in video'
                    }
                
                # Generate embeddings for best frames
                embeddings = []
                best_quality_score = 0
                
                # Sort by quality and take top 5 frames
                sorted_frames = sorted(
                    zip(high_quality_frames, quality_scores), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                for frame_data, quality_score in sorted_frames:
                    try:
                        # Generate embedding for this frame
                        embedding_result = FaceRecognitionService.generate_embedding(frame_data)
                        
                        if embedding_result['success']:
                            embeddings.append(embedding_result['embedding'])
                            best_quality_score = max(best_quality_score, quality_score)
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for frame: {str(e)}")
                        continue
                
                if not embeddings:
                    return {
                        'success': False,
                        'error': 'Failed to generate embeddings from video frames'
                    }
                
                # Create master embedding (average of all embeddings)
                master_embedding = np.mean(embeddings, axis=0)
                master_embedding = master_embedding / np.linalg.norm(master_embedding)  # Normalize
                
                # Update student record with embedding
                student.master_embedding = master_embedding.tobytes()
                student.face_quality_score = best_quality_score
                student.enrollment_video_path = f"student_{student_id}_enrollment.mp4"
                
                db.session.commit()
                
                return {
                    'success': True,
                    'quality_score': best_quality_score,
                    'frames_processed': len(high_quality_frames),
                    'embedding_dimension': len(master_embedding),
                    'message': 'Video processed and master embedding created successfully'
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Video enrollment processing error: {str(e)}")
            return {
                'success': False,
                'error': f'Video processing failed: {str(e)}'
            }
