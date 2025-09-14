from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.face_recognition_service import FaceRecognitionService
import logging
import base64

face_recognition_bp = Blueprint('face_recognition', __name__)
logger = logging.getLogger(__name__)

@face_recognition_bp.route('/detect', methods=['POST'])
@jwt_required()
def detect_faces():
    """Detect faces in image/video frame"""
    try:
        data = request.get_json()
        
        if 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_data'])
        
        # Detect faces
        detection_result = FaceRecognitionService.detect_faces(image_data)
        
        return jsonify({
            'faces_detected': len(detection_result['faces']),
            'faces': detection_result['faces'],
            'processing_time': detection_result['processing_time']
        }), 200
        
    except Exception as e:
        logger.error(f"Face detection error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/embedding', methods=['POST'])
@jwt_required()
def generate_embedding():
    """Generate face embedding for enrollment"""
    try:
        data = request.get_json()
        
        if 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_data'])
        
        # Generate embedding
        embedding_result = FaceRecognitionService.generate_embedding(image_data)
        
        if not embedding_result['success']:
            return jsonify({'error': embedding_result['message']}), 400
        
        return jsonify({
            'embedding': embedding_result['embedding'].tolist(),
            'quality_score': embedding_result['quality_score'],
            'face_coordinates': embedding_result['face_coordinates']
        }), 200
        
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/recognize', methods=['POST'])
@jwt_required()
def recognize_face():
    """Recognize face against database"""
    try:
        data = request.get_json()
        
        required_fields = ['image_data', 'class_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_data'])
        
        # Recognize face
        recognition_result = FaceRecognitionService.recognize_face(
            image_data=image_data,
            class_id=data['class_id'],
            threshold=data.get('threshold', 0.6)
        )
        
        if recognition_result['student_found']:
            return jsonify({
                'student_found': True,
                'student': recognition_result['student'],
                'confidence_score': recognition_result['confidence_score'],
                'face_coordinates': recognition_result['face_coordinates']
            }), 200
        else:
            return jsonify({
                'student_found': False,
                'max_confidence': recognition_result.get('max_confidence', 0),
                'message': 'No matching student found'
            }), 404
        
    except Exception as e:
        logger.error(f"Face recognition error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/quality', methods=['POST'])
@jwt_required()
def check_face_quality():
    """Check face image quality for enrollment"""
    try:
        data = request.get_json()
        
        if 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_data'])
        
        # Check quality
        quality_result = FaceRecognitionService.check_face_quality(image_data)
        
        return jsonify({
            'quality_score': quality_result['quality_score'],
            'is_good_quality': quality_result['is_good_quality'],
            'issues': quality_result['issues'],
            'recommendations': quality_result['recommendations']
        }), 200
        
    except Exception as e:
        logger.error(f"Face quality check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/liveness', methods=['POST'])
@jwt_required()
def check_liveness():
    """Check if face is from a live person (anti-spoofing)"""
    try:
        data = request.get_json()
        
        if 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image_data'])
        
        # Check liveness
        liveness_result = FaceRecognitionService.check_liveness(image_data)
        
        return jsonify({
            'is_live': liveness_result['is_live'],
            'confidence': liveness_result['confidence'],
            'anti_spoofing_score': liveness_result['anti_spoofing_score']
        }), 200
        
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/video_embedding', methods=['POST'])
@jwt_required()
def generate_embedding_from_video():
    """Generate face embedding from video for student enrollment"""
    try:
        data = request.get_json()
        
        required_fields = ['student_id', 'video_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        student_id = data['student_id']
        video_data = base64.b64decode(data['video_data'])
        
        # Process video and generate embeddings
        embedding_result = FaceRecognitionService.process_video_enrollment(
            student_id=student_id,
            video_data=video_data
        )
        
        if embedding_result['success']:
            return jsonify({
                'success': True,
                'quality_score': embedding_result['quality_score'],
                'frames_processed': embedding_result['frames_processed'],
                'embedding_dimension': embedding_result['embedding_dimension'],
                'message': 'Video processed successfully and embeddings generated'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': embedding_result['error']
            }), 400
        
    except Exception as e:
        logger.error(f"Video embedding generation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/save_frame', methods=['POST'])
def save_frame():
    """Save individual frame to database during enrollment"""
    try:
        data = request.get_json()
        
        required_fields = ['student_name', 'roll_number', 'frame_number', 'image_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract data
        student_name = data['student_name']
        roll_number = data['roll_number']
        frame_number = data['frame_number']
        image_data = data['image_data']
        timestamp = data.get('timestamp', None)
        
        # Save frame to database/storage
        result = FaceRecognitionService.save_enrollment_frame(
            student_name=student_name,
            roll_number=roll_number,
            frame_number=frame_number,
            image_data=image_data,
            timestamp=timestamp
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'frame_id': result.get('frame_id'),
                'storage_path': result.get('storage_path'),
                'message': f'Frame {frame_number} saved successfully for {student_name} ({roll_number})'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
    except Exception as e:
        logger.error(f"Frame saving error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@face_recognition_bp.route('/process_enrollment_frames', methods=['POST'])
def process_enrollment_frames():
    """Process all saved enrollment frames to create master embedding"""
    try:
        data = request.get_json()
        
        required_fields = ['student_id', 'student_name', 'roll_number']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        student_id = data['student_id']
        student_name = data['student_name']
        roll_number = data['roll_number']
        frame_count = data.get('frame_count', 0)
        
        # Process the saved frames for this student
        result = FaceRecognitionService.process_saved_enrollment_frames(
            student_id=student_id,
            student_name=student_name,
            roll_number=roll_number
        )
        
        if result['success']:
            return jsonify({
                'success': True,
                'quality_score': result.get('quality_score', 95),
                'frames_processed': result.get('frames_processed', frame_count),
                'embedding_dimension': result.get('embedding_dimension', 512),
                'face_detection_success': result.get('face_detection_success', True),
                'message': f'Enrollment frames processed successfully for {student_name}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
    except Exception as e:
        logger.error(f"Enrollment frame processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
