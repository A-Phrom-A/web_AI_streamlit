import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import joblib

# อ่านข้อมูล
apple_df = pd.read_csv("apple_mobile_price.csv", encoding='latin1')
non_apple_df = pd.read_csv("non_apple_mobile_price.csv", encoding='latin1')

# กรองเฉพาะปี 2020-2025
apple_df = apple_df[apple_df["Launched Year"].astype(str).str[:4].astype(float).between(2020, 2025)].copy()
non_apple_df = non_apple_df[non_apple_df["Launched Year"].astype(str).str[:4].astype(float).between(2020, 2025)].copy()

# เติมค่า NaN ด้วยค่าเฉลี่ย
apple_df.fillna(apple_df.mean(numeric_only=True), inplace=True)
non_apple_df.fillna(non_apple_df.mean(numeric_only=True), inplace=True)

# ฟังก์ชันทำความสะอาดข้อมูล
def clean_battery(value):
    return float(re.sub(r"[^0-9]", "", str(value))) if pd.notnull(value) else np.nan

def clean_ram(value):
    return int(re.sub(r"[^0-9]", "", str(value))) if pd.notnull(value) else np.nan

def clean_screen_size(value):
    values = [float(s) for s in re.findall(r"\d+\.?\d*", str(value))]
    return max(values) if values else np.nan

def clean_camera(value):
    values = [float(v) for v in re.findall(r"\d+\.?\d*", str(value))]
    return max(values) if values else np.nan

# ทำความสะอาดข้อมูล
for df in [apple_df, non_apple_df]:
    df["Battery Capacity"] = df["Battery Capacity"].apply(clean_battery)
    df["RAM"] = df["RAM"].apply(clean_ram)
    df["Screen Size"] = df["Screen Size"].apply(clean_screen_size)
    df["Front Camera"] = df["Front Camera"].apply(clean_camera)
    df["Back Camera"] = df["Back Camera"].apply(clean_camera)

# เลือกฟีเจอร์และตัวแปรเป้าหมาย
features = ['RAM', 'Battery Capacity', 'Screen Size', 'Front Camera', 'Back Camera', 'Processor', 'Company Name']

# ฟังก์ชันแปลงข้อมูลหมวดหมู่
def encode_features(df):
    label_encoders = {}
    
    # สำหรับคอลัมน์ Processor และ Company Name
    for col in ["Processor", "Company Name"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# แปลงข้อมูล
apple_df, apple_encoders = encode_features(apple_df)
joblib.dump(apple_encoders, "apple_label_encoders.pkl")

non_apple_df, non_apple_encoders = encode_features(non_apple_df)
joblib.dump(non_apple_encoders, "non_apple_label_encoders.pkl")

# แยกข้อมูล
X_apple, y_apple = apple_df[features], apple_df["Launched Price (USA)"]
X_non_apple, y_non_apple = non_apple_df[features], non_apple_df["Launched Price (USA)"]

X_train_apple, X_test_apple, y_train_apple, y_test_apple = train_test_split(X_apple, y_apple, test_size=0.2, random_state=42)
X_train_non_apple, X_test_non_apple, y_train_non_apple, y_test_non_apple = train_test_split(X_non_apple, y_non_apple, test_size=0.2, random_state=42)

# สร้างโมเดล XGBoost
apple_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=5, random_state=42)
non_apple_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=5, random_state=42)

# ฝึกสอนโมเดล
apple_model.fit(X_train_apple, y_train_apple)
non_apple_model.fit(X_train_non_apple, y_train_non_apple)

# บันทึกโมเดล
joblib.dump(apple_model, "apple_xgboost_model.pkl")
joblib.dump(non_apple_model, "non_apple_xgboost_model.pkl")

print("Models and encoders saved successfully!")

# ฟังก์ชันทำนายราคาจากบริษัท
def predict_price(sample_data):
    df = pd.DataFrame(sample_data)
    
    # ดึงชื่อบริษัท
    company = df.iloc[0]['Company Name']
    
    # เลือกโมเดลและ encoder ที่เหมาะสม
    if company == 'Apple':
        model = apple_model
        encoders = apple_encoders
    else:
        model = non_apple_model
        encoders = non_apple_encoders
    
    # แปลงค่าของ Processor โดยใช้ LabelEncoder ของบริษัทนั้น ๆ
    df["Processor"] = df["Processor"].apply(lambda x: encoders["Processor"].transform([x])[0] if x in encoders["Processor"].classes_ else 0)
    
    # แปลงค่าของ Company Name โดยใช้ LabelEncoder ของบริษัทนั้น ๆ
    df["Company Name"] = encoders["Company Name"].transform(df["Company Name"])

    # เลือกฟีเจอร์
    X_sample = df[features]
    
    # ทำนายราคา
    return model.predict(X_sample)

# ตัวอย่างการทำนาย
sample_phones = [
    {'RAM': 6, 'Battery Capacity': 4000, 'Screen Size': 6.7, 'Front Camera': 20, 'Back Camera': 42, 'Company Name': 'Apple', 'Processor': 'Apple A17'},
    {'RAM': 8, 'Battery Capacity': 5000, 'Screen Size': 6.7, 'Front Camera': 20, 'Back Camera': 64, 'Company Name': 'Samsung', 'Processor': 'Snapdragon'}
]

for i, phone in enumerate(sample_phones):
    print(f"Prediction for phone {i+1}: ${predict_price([phone])[0]:.2f}")

