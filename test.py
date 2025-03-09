import joblib

# โหลด label encoders
apple_encoders = joblib.load("apple_label_encoders.pkl")
non_apple_encoders = joblib.load("non_apple_label_encoders.pkl")

# แสดงค่าของ label encoder สำหรับแต่ละฟีเจอร์
for feature, encoder in apple_encoders.items():
    print(f"Apple - {feature} encoder classes: {encoder.classes_}")

for feature, encoder in non_apple_encoders.items():
    print(f"Non-Apple - {feature} encoder classes: {encoder.classes_}")
