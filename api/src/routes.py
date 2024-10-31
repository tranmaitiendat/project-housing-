from flask import Blueprint, request, jsonify
import pandas as pd
import pickle

api = Blueprint('api', __name__)

# Dữ liệu mẫu cho người dùng
users = {
    "test@example.com": "password123"
}

# Tải mô hình đã huấn luyện
try:
    with open('/Users/dlb/Documents/Zalo Received Files/project/api/model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Model file not found. Ensure 'model.pkl' is in the 'api' directory.")

@api.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email in users and users[email] == password:
        return jsonify({"message": "Login successful!"}), 200
    else:
        return jsonify({"message": "Invalid email or password!"}), 401

@api.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if email in users:
        return jsonify({"message": "Email already registered!"}), 409  # Conflict

    users[email] = password
    return jsonify({"message": "User registered successfully!"}), 201

@api.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    # Kiểm tra tính hợp lệ của dữ liệu đầu vào
    required_features = ['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car', 'Landsize']
    features = []
    for feature in required_features:
        value = data.get(feature)
        if value is None:
            return jsonify({"error": f"Missing feature: {feature}"}), 400
        features.append(value)

    # Dự đoán giá nhà
    try:
        prediction = model.predict([features])
        return jsonify({"predicted_price": prediction[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/users', methods=['GET'])
def get_users():
    return jsonify({"users": list(users.keys())}), 200
