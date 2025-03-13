# การใช้งานของแต่ละ dataset

## dataset หลักสำหรับการ เทรน Predict New_Mobile_Price(2020-2025) ใช้จาก `mobile_data_corrected.csv`
  แต่เพื่อป้องกัน การ bias ที่ส่งผลมาจาก ram ที่น้อยของ Apple เลยต้องแยกเทรนออกมาเป็น 2 model

1. `apple_xgboost_model.pkl` ใช้ **dataset** จาก `mobile_data_corrected.csv` แล้ว split ออกมาแค่ Apple ได้ไฟล์ dataset : `apple_mobile_price.csv`

2. `non_apple_xgboost_model.pkl` ใช้ **dataset** จาก `mobile_data_corrected.csv` แล้ว split ทีเหลือออกมา ได้ไฟล์ dataset : `non_apple_mobile_price.csv`

## dataset หลักสำหรับการ เทรน Predict_Old_Mobile_Price(2000-2010) ใช้จาก train.csv

  โดยมีการใช้ dataset : `test.csv` สำหรับการตรวจสอบค่าความถูกต้อง
และเก็บ ผลลัพะ์ในการ เทรนลงในไฟล์ `predicted_test_results.csv`
