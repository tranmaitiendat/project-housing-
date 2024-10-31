# src/utils.py

def validate_input(data):
    """
    Hàm kiểm tra tính hợp lệ của đầu vào.
    """
    required_fields = ['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car', 'Landsize']
    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"
    return True, ""
