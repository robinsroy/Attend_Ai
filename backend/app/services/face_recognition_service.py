import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import base64
import os
from PIL import Image
import io

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Service for face detection, recognition, and quality assessment"""
    
    @staticmethod
    def detect_faces(image_data: bytes) -> Dict[str, Any]:
        """Detect faces in image data"""
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
