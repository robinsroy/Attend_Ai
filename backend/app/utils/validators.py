def validate_login_data(data: dict) -> tuple[bool, list]:
    """Validate login request data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    # Required fields
    required_fields = ['username', 'password']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Username validation
    if 'username' in data:
        username = data['username'].strip()
        if len(username) < 3:
            errors.append("Username must be at least 3 characters long")
        if len(username) > 50:
            errors.append("Username cannot exceed 50 characters")
    
    # Password validation
    if 'password' in data:
        password = data['password']
        if len(password) < 6:
            errors.append("Password must be at least 6 characters long")
    
    return len(errors) == 0, errors

def validate_student_data(data: dict) -> tuple[bool, list]:
    """Validate student enrollment data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    # Required fields
    required_fields = ['roll_number', 'full_name', 'class_id']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Roll number validation
    if 'roll_number' in data:
        roll_number = str(data['roll_number']).strip()
        if len(roll_number) < 3:
            errors.append("Roll number must be at least 3 characters long")
        if len(roll_number) > 20:
            errors.append("Roll number cannot exceed 20 characters")
    
    # Full name validation
    if 'full_name' in data:
        full_name = data['full_name'].strip()
        if len(full_name) < 2:
            errors.append("Full name must be at least 2 characters long")
        if len(full_name) > 100:
            errors.append("Full name cannot exceed 100 characters")
    
    # Class ID validation
    if 'class_id' in data:
        try:
            class_id = int(data['class_id'])
            if class_id <= 0:
                errors.append("Class ID must be a positive integer")
        except (ValueError, TypeError):
            errors.append("Class ID must be a valid integer")
    
    # Optional phone validation
    if 'guardian_phone' in data and data['guardian_phone']:
        phone = str(data['guardian_phone']).strip()
        if len(phone) < 10 or len(phone) > 15:
            errors.append("Guardian phone must be between 10-15 characters")
    
    return len(errors) == 0, errors

def validate_attendance_session_data(data: dict) -> tuple[bool, list]:
    """Validate attendance session start data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    # Required fields
    required_fields = ['class_id', 'period_id']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Class ID validation
    if 'class_id' in data:
        try:
            class_id = int(data['class_id'])
            if class_id <= 0:
                errors.append("Class ID must be a positive integer")
        except (ValueError, TypeError):
            errors.append("Class ID must be a valid integer")
    
    # Period ID validation
    if 'period_id' in data:
        try:
            period_id = int(data['period_id'])
            if period_id <= 0:
                errors.append("Period ID must be a positive integer")
        except (ValueError, TypeError):
            errors.append("Period ID must be a valid integer")
    
    # Optional threshold validation
    if 'detection_threshold' in data and data['detection_threshold'] is not None:
        try:
            threshold = float(data['detection_threshold'])
            if threshold < 0.1 or threshold > 1.0:
                errors.append("Detection threshold must be between 0.1 and 1.0")
        except (ValueError, TypeError):
            errors.append("Detection threshold must be a valid number")
    
    return len(errors) == 0, errors

def validate_class_data(data: dict) -> tuple[bool, list]:
    """Validate class creation data"""
    errors = []
    
    if not data:
        errors.append("No data provided")
        return False, errors
    
    # Required fields
    required_fields = ['name', 'section', 'grade', 'subject', 'academic_year']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Name validation
    if 'name' in data:
        name = data['name'].strip()
        if len(name) < 3:
            errors.append("Class name must be at least 3 characters long")
        if len(name) > 100:
            errors.append("Class name cannot exceed 100 characters")
    
    # Section validation
    if 'section' in data:
        section = data['section'].strip()
        if len(section) > 10:
            errors.append("Section cannot exceed 10 characters")
    
    # Grade validation
    if 'grade' in data:
        grade = data['grade'].strip()
        if len(grade) > 10:
            errors.append("Grade cannot exceed 10 characters")
    
    return len(errors) == 0, errors

def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
