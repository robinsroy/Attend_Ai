from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.class_service import ClassService
from app.services.attendance_service import AttendanceService
import logging

classes_bp = Blueprint('classes', __name__)
logger = logging.getLogger(__name__)

@classes_bp.route('/', methods=['GET'])
@jwt_required()
def get_classes():
    """Get all classes"""
    try:
        classes = ClassService.get_all_classes()
        
        return jsonify({
            'classes': [cls.to_dict() for cls in classes],
            'count': len(classes)
        }), 200
        
    except Exception as e:
        logger.error(f"Get classes error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@classes_bp.route('/<int:class_id>', methods=['GET'])
@jwt_required()
def get_class(class_id):
    """Get specific class by ID"""
    try:
        class_obj = ClassService.get_class_by_id(class_id)
        
        if not class_obj:
            return jsonify({'error': 'Class not found'}), 404
        
        return jsonify({
            'class': class_obj.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Get class error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@classes_bp.route('/<int:class_id>/students', methods=['GET'])
@jwt_required()
def get_class_students(class_id):
    """Get all students in a specific class"""
    try:
        class_obj = ClassService.get_class_by_id(class_id)
        if not class_obj:
            return jsonify({'error': 'Class not found'}), 404
        
        students = ClassService.get_class_students(class_id)
        
        return jsonify({
            'class_id': class_id,
            'class_name': class_obj.name,
            'students': [student.to_dict() for student in students],
            'total_students': len(students)
        }), 200
        
    except Exception as e:
        logger.error(f"Get class students error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@classes_bp.route('/<int:class_id>/periods', methods=['GET'])
@jwt_required()
def get_class_periods(class_id):
    """Get all periods for a specific class"""
    try:
        class_obj = ClassService.get_class_by_id(class_id)
        if not class_obj:
            return jsonify({'error': 'Class not found'}), 404
        
        periods = ClassService.get_class_periods(class_id)
        
        return jsonify({
            'class_id': class_id,
            'class_name': class_obj.name,
            'periods': [period.to_dict() for period in periods]
        }), 200
        
    except Exception as e:
        logger.error(f"Get class periods error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@classes_bp.route('/', methods=['POST'])
@jwt_required()
def create_class():
    """Create new class"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # Basic validation
        required_fields = ['name', 'section', 'grade', 'subject', 'academic_year']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        class_obj = ClassService.create_class(data)
        
        logger.info(f"Class {class_obj.name} created by user {current_user_id}")
        
        return jsonify({
            'message': 'Class created successfully',
            'class': class_obj.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Create class error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@classes_bp.route('/<int:class_id>/periods', methods=['POST'])
@jwt_required()
def create_period(class_id):
    """Create new period for a class"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        class_obj = ClassService.get_class_by_id(class_id)
        if not class_obj:
            return jsonify({'error': 'Class not found'}), 404
        
        # Basic validation
        required_fields = ['period_number', 'period_name', 'start_time', 'end_time', 'days_of_week']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        data['class_id'] = class_id
        period = ClassService.create_period(data)
        
        logger.info(f"Period {period.period_name} created for class {class_id} by user {current_user_id}")
        
        return jsonify({
            'message': 'Period created successfully',
            'period': period.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Create period error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
