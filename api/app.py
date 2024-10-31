from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from flask_bcrypt import Bcrypt
import sqlite3

app = Flask(__name__)
CORS(app)  # Cho phép CORS từ tất cả các nguồn
bcrypt = Bcrypt(app)

# Kết nối và thiết lập SQLite
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Đọc và xử lý dữ liệu
data = pd.read_csv('cleaned_data.csv')
data.replace('missing', pd.NA, inplace=True)
numerical_features = ['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car',
                      'Landsize', 'BuildingArea', 'YearBuilt', 'Postcode']
data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')

X = data[['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car',
          'Landsize', 'BuildingArea', 'YearBuilt', 'Postcode',
          'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']]
y = data['Price']

categorical_features = ['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']

# Chuẩn bị mô hình và xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', LinearRegression())])
model_pipeline.fit(X, y)

# Lưu mô hình đã được huấn luyện
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Tải lại mô hình
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Hàm xử lý đầu vào
def preprocess_input(data):
    required_columns = ['Rooms', 'Distance', 'Bedroom', 'Bathroom', 'Car',
                        'Landsize', 'BuildingArea', 'YearBuilt', 'Postcode',
                        'Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
    input_data = data[required_columns]
    return input_data

# Hàm lưu người dùng vào SQLite
def add_user_to_db(username, hashed_password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Kiểm tra người dùng tồn tại trong SQLite
def get_user_from_db(username):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Endpoint đăng ký
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if get_user_from_db(username):
        return jsonify({'message': 'User already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    add_user_to_db(username, hashed_password)
    return jsonify({'message': 'User registered successfully'}), 201

# Endpoint đăng nhập
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = get_user_from_db(username)
    if not user or not bcrypt.check_password_hash(user[2], password):  # user[2] là trường password
        return jsonify({'message': 'Invalid username or password'}), 401
    return jsonify({'message': 'Login successful'}), 200

# Endpoint dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    processed_data = preprocess_input(input_df)

    prediction = model.predict(processed_data)
    return jsonify({'predicted_price': prediction[0]})

# API: Phân phối giá nhà
@app.route('/price-distribution', methods=['GET'])
def price_distribution():
    price_data = data['Price'].dropna().tolist()
    return jsonify(price_data)

# API: Giá nhà theo số lượng phòng
@app.route('/price-by-rooms', methods=['GET'])
def price_by_rooms():
    grouped_data = data.groupby('Rooms')['Price'].apply(list).to_dict()
    return jsonify(grouped_data)

# API: Diện tích đất và giá nhà (biểu đồ phân tán)
@app.route('/landsize-vs-price', methods=['GET'])
def landsize_vs_price():
    scatter_data = data[['Landsize', 'Price']].dropna().to_dict(orient='records')
    return jsonify(scatter_data)

# API: Thống kê trung bình giá nhà theo khu vực
@app.route('/average-price-by-region', methods=['GET'])
def average_price_by_region():
    avg_price_by_region = data.groupby('Regionname')['Price'].mean().dropna().to_dict()
    return jsonify(avg_price_by_region)

# API: Số lượng bất động sản theo loại hình
@app.route('/property-count-by-type', methods=['GET'])
def property_count_by_type():
    property_counts = data['Type'].value_counts().to_dict()
    return jsonify(property_counts)

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
