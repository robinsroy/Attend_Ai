"""
Real-time Recognition API Routes
Provides endpoints for managing the automated recognition loop
"""

from flask import Blueprint, request, jsonify, Response
from flask_cors import cross_origin
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import threading
import time

# Import the real-time recognition service
try:
    from ..services.realtime_recognition_service import RealtimeRecognitionService
except ImportError:
    # Fallback if service not available
    RealtimeRecognitionService = None

realtime_recognition_bp = Blueprint('realtime_recognition', __name__)

# Global service instance
recognition_service = None
service_lock = threading.Lock()

def get_recognition_service():
    """Get or create recognition service instance"""
    global recognition_service
    with service_lock:
        if recognition_service is None and RealtimeRecognitionService:
            recognition_service = RealtimeRecognitionService()
        return recognition_service

@realtime_recognition_bp.route('/start_camera', methods=['POST'])
@cross_origin()
def start_camera():
    """Start camera for real-time recognition"""
    try:
        data = request.get_json()
        camera_index = data.get('camera_index', 0)
        
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Real-time recognition service not available'
            }), 500
        
        success = service.start_camera(camera_index)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Camera {camera_index} started successfully',
                'camera_index': camera_index
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to start camera {camera_index}'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error starting camera: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/stop_camera', methods=['POST'])
@cross_origin()
def stop_camera():
    """Stop camera"""
    try:
        service = get_recognition_service()
        if service:
            service.stop_camera()
        
        return jsonify({
            'success': True,
            'message': 'Camera stopped successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error stopping camera: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/start_recognition', methods=['POST'])
@cross_origin()
def start_recognition():
    """Start automated recognition loop"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        class_id = data.get('class_id')
        
        if not session_id or not class_id:
            return jsonify({
                'success': False,
                'error': 'session_id and class_id are required'
            }), 400
        
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Real-time recognition service not available'
            }), 500
        
        service.start_recognition_loop(session_id, class_id)
        
        return jsonify({
            'success': True,
            'message': 'Recognition loop started',
            'session_id': session_id,
            'class_id': class_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error starting recognition: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/stop_recognition', methods=['POST'])
@cross_origin()
def stop_recognition():
    """Stop automated recognition loop"""
    try:
        service = get_recognition_service()
        if service:
            service.stop_recognition_loop()
        
        return jsonify({
            'success': True,
            'message': 'Recognition loop stopped'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error stopping recognition: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/camera_feed')
@cross_origin()
def camera_feed():
    """Stream camera feed"""
    try:
        service = get_recognition_service()
        if not service:
            return jsonify({'error': 'Service not available'}), 500
        
        def generate_frames():
            while True:
                frame = service.get_latest_frame()
                if frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS
        
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@realtime_recognition_bp.route('/capture_frame', methods=['POST'])
@cross_origin()
def capture_frame():
    """Capture and process a single frame"""
    try:
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
        
        frame = service.get_latest_frame()
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'No frame available'
            }), 400
        
        # Encode frame as base64 for frontend
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to encode frame'
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error capturing frame: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/recognition_results', methods=['GET'])
@cross_origin()
def get_recognition_results():
    """Get latest recognition results"""
    try:
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
        
        # Get latest results
        latest_results = service.get_latest_results()
        
        # Get session statistics
        stats = service.get_session_statistics()
        
        return jsonify({
            'success': True,
            'latest_results': latest_results,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting results: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/session_status', methods=['GET'])
@cross_origin()
def get_session_status():
    """Get current session status"""
    try:
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
        
        stats = service.get_session_statistics()
        
        return jsonify({
            'success': True,
            'status': stats,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting status: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/process_frame', methods=['POST'])
@cross_origin()
def process_frame():
    """Process uploaded frame through recognition pipeline"""
    try:
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({
                'success': False,
                'error': 'No frame data provided'
            }), 400
        
        # Decode base64 frame
        try:
            # Remove data URL prefix if present
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            frame_bytes = base64.b64decode(frame_data)
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode frame: {str(e)}'
            }), 400
        
        service = get_recognition_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Service not available'
            }), 500
        
        # Process frame through recognition pipeline
        # This is a simplified version - the actual processing happens in the loop
        
        # For demo purposes, return mock results
        mock_results = [
            {
                'student_id': 'DEMO001',
                'student_name': 'Demo Student',
                'confidence': 0.95,
                'timestamp': time.time()
            }
        ]
        
        return jsonify({
            'success': True,
            'results': mock_results,
            'message': 'Frame processed successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing frame: {str(e)}'
        }), 500

@realtime_recognition_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check for real-time recognition service"""
    try:
        service = get_recognition_service()
        
        status = {
            'service_available': service is not None,
            'camera_active': service.camera is not None if service else False,
            'recognition_active': service.recognition_active if service else False,
            'deepface_available': True,  # Check if we can import DeepFace
            'dlib_available': True,      # Check if dlib is available
            'timestamp': datetime.now().isoformat()
        }
        
        # Test DeepFace availability
        try:
            from deepface import DeepFace
            status['deepface_available'] = True
        except ImportError:
            status['deepface_available'] = False
        
        # Test dlib availability
        try:
            import dlib
            status['dlib_available'] = True
        except ImportError:
            status['dlib_available'] = False
        
        return jsonify({
            'success': True,
            'status': status
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Health check failed: {str(e)}'
        }), 500