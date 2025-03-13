import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np




# โหลดโมเดล Neural Network
nn_model = tf.keras.models.load_model('mobile_price_model.keras')  # โหลดโมเดล Neural Network ด้วย Keras

non_apple_model = joblib.load("non_apple_xgboost_model.pkl")
label_encoders_non_apple_model  = joblib.load("non_apple_label_encoders.pkl")

apple_model = joblib.load("apple_xgboost_model.pkl")
label_encoders_apple_model  = joblib.load("apple_label_encoders.pkl")

def predict_mobile_price_xgb(company_name, ram, front_cam, back_cam, processor, battery, screen):
    st.info("กำลังประมวลผลข้อมูล...")
    time.sleep(1)  # ให้ความรู้สึกว่าระบบกำลังทำงาน

    sample_data = {
        'Company Name': [company_name],
        'RAM': [ram],
        'Front Camera': [front_cam],
        'Back Camera': [back_cam],
        'Processor': [processor],
        'Battery Capacity': [battery],
        'Screen Size': [screen]
    }
    df = pd.DataFrame(sample_data)

    expected_columns = ['RAM', 'Battery Capacity', 'Screen Size', 'Front Camera', 'Back Camera', 'Processor', 'Company Name']
    df = df[expected_columns]

    model = apple_model if company_name.lower() == 'apple' else non_apple_model
    encoders = label_encoders_apple_model if company_name.lower() == 'apple' else label_encoders_non_apple_model
    
    try:
        df["Company Name"] = encoders["Company Name"].transform(df["Company Name"])
    except KeyError:
        st.error("Encoder สำหรับ 'Company Name' ไม่พบใน label encoders")
        return
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลง 'Company Name': {str(e)}")
        return

    if "Processor" in encoders:
        df["Processor"] = df["Processor"].apply(
            lambda x: encoders["Processor"].transform([x])[0] if x in encoders["Processor"].classes_ else 0
        )
    else:
        st.error("Processor encoder not found in label encoders.")
        return None

    X_sample = df[expected_columns]
    progress_bar = st.progress(0)
    for percent_complete in range(1, 101):
        time.sleep(0.01)
        progress_bar.progress(percent_complete)

    price_predictions = model.predict(X_sample)
    st.write(f"Prediction: {price_predictions}")


    # แสดงกราฟกระจายผลลัพธ์การทำนาย
    st.subheader("📊 Price Distribution")

    # สุ่มตัวอย่างราคาจากการกระจายปกติรอบราคาที่ทำนายได้
    sample_prices = np.random.normal(price_predictions[0], 2000, 100)  # เพิ่มจำนวนตัวอย่างเป็น 100

    # กำหนดช่วงราคาที่แสดงบนแกน x
    min_price = max(0, int(min(sample_prices) // 500 * 500))  # ปรับให้เป็นช่วงละ 500
    max_price = int(max(sample_prices) // 500 * 500 + 500)  
    bins = np.arange(min_price, max_price + 500, 500)  # สร้าง bin ละ 500 บาท

    # สร้างกราฟ
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#f4f4f4")  # เพิ่มพื้นหลังสีเทาอ่อน
    ax.set_facecolor("#eaeaea")  # เพิ่มสีพื้นหลังของ plot

    # ฮิสโตแกรมที่ละเอียดขึ้น
    sns.histplot(sample_prices, bins=bins, kde=True, color="#3498db", alpha=0.8, edgecolor="black", ax=ax)
    ax.axvline(price_predictions[0], color='red', linestyle='--', linewidth=2, label=f'Predicted Price: {price_predictions[0]:,.0f} บาท')

    # เพิ่มตัวเลขด้านบนแท่ง Histogram
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2, height + 0.5, f'{int(height)}', ha='center', fontsize=10, fontweight="bold", color="black")

    # ตั้งค่าป้ายกำกับแกน x
    tick_labels = [f"{int(b):,}" for b in bins[:-1]]
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=11, fontweight="bold", fontname="Arial")

    # ตั้งค่าป้ายแกน y
    ax.set_yticklabels(ax.get_yticks(), fontsize=11, fontweight="bold", fontname="Arial")

    # เพิ่ม Gridlines
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # เพิ่ม annotation ที่เส้นราคาทำนายได้
    ax.annotate(f'Predicted: {price_predictions[0]:,.0f} บาท',
                xy=(price_predictions[0], ax.get_ylim()[1]*0.8), 
                xytext=(price_predictions[0] + 1000, ax.get_ylim()[1]*0.9),
                arrowprops=dict(facecolor='red', arrowstyle='->', lw=2),
                fontsize=12, fontweight="bold", color="red")

    # ปรับแต่งเพิ่มเติม
    ax.set_title("📈 Price Distribution", fontsize=14, fontweight="bold", fontname="Arial")
    ax.set_xlabel("Price (Baht)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.legend()
    sns.despine()

    st.pyplot(fig)

    return price_predictions[0]



import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ฟังก์ชันในการทำนายราคามือถือด้วย Neural Network
def predict_mobile_price_nn(battery_power, blue, clock_speed, dual_sim, fc, four_g, 
                             int_memory, m_dep, mobile_wt, n_cores, pc, px_height, 
                             px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi):
    input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, 
                            int_memory, m_dep, mobile_wt, n_cores, pc, px_height, 
                            px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])  # ขนาดข้อมูลเป็น (1, 20)

    # ทำนายผล
    prediction = nn_model.predict(input_data)
    price_predicted = prediction[0][0]

    # แสดงกราฟกระจายผลลัพธ์การทำนาย
    st.subheader("📊 Price Distribution")

    # สุ่มตัวอย่างราคาจากการกระจายปกติรอบราคาที่ทำนายได้
    sample_prices = np.random.normal(price_predicted, 2000, 100)  # เพิ่มจำนวนตัวอย่างเป็น 100

    # กำหนดช่วงราคาที่แสดงบนแกน x
    min_price = max(0, int(min(sample_prices) // 500 * 500))  # ปรับให้เป็นช่วงละ 500
    max_price = int(max(sample_prices) // 500 * 500 + 500)  
    bins = np.arange(min_price, max_price + 500, 500)  # สร้าง bin ละ 500 บาท

    # สร้างกราฟ
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#f4f4f4")  # เพิ่มพื้นหลังสีเทาอ่อน
    ax.set_facecolor("#eaeaea")  # เพิ่มสีพื้นหลังของ plot

    # ฮิสโตแกรมที่ละเอียดขึ้น
    sns.histplot(sample_prices, bins=bins, kde=True, color="#3498db", alpha=0.8, edgecolor="black", ax=ax)
    ax.axvline(price_predicted, color='red', linestyle='--', linewidth=2, label=f'Predicted Price: {price_predicted:,.0f} บาท')

    # เพิ่มตัวเลขด้านบนแท่ง Histogram
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2, height + 0.5, f'{int(height)}', ha='center', fontsize=10, fontweight="bold", color="black")

    # ตั้งค่าป้ายกำกับแกน x
    tick_labels = [f"{int(b):,}" for b in bins[:-1]]
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=11, fontweight="bold", fontname="Arial")

    # ตั้งค่าป้ายแกน y
    ax.set_yticklabels(ax.get_yticks(), fontsize=11, fontweight="bold", fontname="Arial")

    # เพิ่ม Gridlines
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # เพิ่ม annotation ที่เส้นราคาทำนายได้
    ax.annotate(f'Predicted: {price_predicted:,.0f} บาท',
                xy=(price_predicted, ax.get_ylim()[1]*0.8), 
                xytext=(price_predicted + 1000, ax.get_ylim()[1]*0.9),
                arrowprops=dict(facecolor='red', arrowstyle='->', lw=2),
                fontsize=12, fontweight="bold", color="red")

    # ปรับแต่งเพิ่มเติม
    ax.set_title("📈 Price Distribution", fontsize=14, fontweight="bold", fontname="Arial")
    ax.set_xlabel("Price (Baht)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.legend()
    sns.despine()

    st.pyplot(fig)

    return price_predicted


# UI หลักของเว็บแอป
def main():
    st.title('Machine Learning & Neural Network Web App')
    menu = ['Home', 'การฝึกฝนโมเดล Machine Learning','Predict New_Mobile_Price(2020-2025)',  'การฝึกฝนโมเดล  Neural Network', 'Predict_Old_Mobile_Price(2000-2010)' , 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Home':
        st.subheader('Project Overview')
        st.write(''' 
            ในโปรเจคนี้เราจะใช้ **Neural Network** และ **XGBoost** เพื่อทำนายราคามือถือโดยพิจารณาจากคุณสมบัติต่างๆ ของมือถือ เช่น หน่วยความจำ, ความเร็วของโปรเซสเซอร์, ขนาดหน้าจอ และอื่นๆ 
            โดยการใช้งาน **Streamlit** จะช่วยให้เราสามารถสร้างและแสดงผลแอปพลิเคชันเว็บที่ใช้งานง่าย 
            ซึ่งผู้ใช้สามารถเลือกดูข้อมูลและทำนายราคามือถือได้แบบอินเทอร์แอคทีฟ

            ### ขั้นตอนในการทำโปรเจค
            1. **เตรียมข้อมูล (Data Preparation)**: เริ่มต้นด้วยการโหลดข้อมูลมือถือจากไฟล์ CSV และทำการแยกข้อมูลเป็นคุณสมบัติ (Features) และเป้าหมาย (Target)
            2. **การฝึกโมเดล (Model Training)**: เราฝึกโมเดล Machine Learning หลายๆ แบบ เช่น Linear Regression, KNN, Logistic Regression, Decision Tree, Random Forest และ Neural Network
            3. **การทดสอบและประเมินผล (Testing & Evaluation)**: เราประเมินประสิทธิภาพของโมเดลด้วยข้อมูลทดสอบ เพื่อเลือกโมเดลที่มีประสิทธิภาพดีที่สุด
            4. **การทำนายราคามือถือ (Mobile Price Prediction)**: สุดท้าย เราใช้โมเดลที่ฝึกเสร็จแล้วในการทำนายราคามือถือในชุดข้อมูลทดสอบ

            ### โมเดลที่ใช้ในโปรเจคนี้
            - **Neural Network**: ใช้ Neural Network ในการทำนายราคามือถือเนื่องจากมันสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนระหว่างข้อมูล
            - **XGBoost**: ใช้ XGBoost เป็นโมเดลที่มีความสามารถในการจัดการข้อมูลที่มีความซับซ้อนสูงและประสิทธิภาพสูง
            - **KNN**: ใช้ KNN เพื่อทำนายราคาจากข้อมูลใกล้เคียง
            - **Random Forest**: ใช้ Random Forest ที่มีการรวมโมเดลหลายๆ ตัวเพื่อให้ได้ผลลัพธ์ที่แม่นยำขึ้น

            ### เทคโนโลยีที่ใช้
            - **Streamlit**: เป็นเครื่องมือสำหรับสร้างเว็บแอปพลิเคชันที่ใช้งานง่าย และรองรับการแสดงผลข้อมูลเชิงโต้ตอบ
            - **XGBoost**: ใช้สำหรับการฝึกโมเดลที่สามารถจัดการกับข้อมูลขนาดใหญ่และซับซ้อนได้
            - **TensorFlow**: ใช้สำหรับการสร้างและฝึก Neural Network
            - **scikit-learn**: ใช้สำหรับโมเดล Machine Learning อื่นๆ เช่น KNN, Random Forest, Logistic Regression เป็นต้น
        ''')

        st.write("โปรเจคนี้เป็นตัวอย่างของการนำ Machine Learning และเว็บแอปพลิเคชันมารวมกัน เพื่อสร้างเครื่องมือที่สามารถทำนายราคามือถือได้อย่างง่ายดาย")
        st.write("คุณสามารถลองเลือกการทำนายราคามือถือจากหน้าอื่นๆ ของแอปนี้")

        # ส่วน Predict New Mobile Price (2020-2025) ที่ใช้ XGBoost
        st.subheader('Predict New Mobile Price (2020-2025) Using XGBoost')
        st.write(''' 
            ในส่วนนี้เราจะใช้ **XGBoost** เพื่อทำนายราคามือถือที่เปิดตัวระหว่างปี 2020 ถึง 2025 โดยใช้ข้อมูลต่างๆ เช่น หน่วยความจำ, ความจุแบตเตอรี่, ขนาดหน้าจอ, กล้องหน้า, กล้องหลัง, และโปรเซสเซอร์
            ซึ่งเราจะมีการเตรียมข้อมูลและฝึกโมเดล XGBoost เพื่อให้สามารถทำนายราคามือถือจากฟีเจอร์ต่างๆ ที่มีอยู่
        ''')
    
    elif choice == 'Predict New_Mobile_Price(2020-2025)':
        st.subheader('ประเมินราคามือถือรุ่นใหม่ (2020-2025)')

        # รับค่าจากผู้ใช้
        company_name = st.selectbox('Company Name', 
                                    ['Apple', 'Samsung', 'OnePlus', 'Vivo', 'iQOO', 'Oppo', 
                                     'Realme', 'Xiaomi', 'Lenovo', 'Motorola', 'Huawei', 
                                     'Nokia', 'Sony', 'Google', 'Tecno', 'Infinix', 'Honor', 'POCO'])

        # แสดงตัวเลือกโปรเซสเซอร์ที่แตกต่างกันตามแบรนด์
        if company_name == 'Apple':
            processor = st.selectbox('Processor', ['Apple A11','Apple A12','Apple A12 Z','Apple A13','Apple A14', 'Apple A15', 'Apple A16','Apple A17','Apple A17 pro','Apple A18','Apple A18 pro'])
        else :
            processor = st.selectbox('Processor', [
                                                        'Exynos', 'Snapdragon', 'MediaTek Helio', 'MediaTek', 'MediaTek Dimensity',
                                                        'Unisoc ', 'Spreadtrum ', 'Qualcomm MSM8916', 'Qualcomm Snapdragon',
                                                        'Dimensity', 'Kirin ', 'Google Tensor'
                                                    ])
       
        ram = st.selectbox('RAM (GB)', [1, 2, 3, 4, 6, 8, 12, 16, 32])
        front_cam = st.number_input('Front Camera (MP)', 1, 50, 10)
        back_cam = st.number_input('Back Camera (MP)', 1, 200, 20)
        
        battery = st.selectbox('Battery Capacity (mAh)', [3000, 4000, 5000, 6000 ,7000 ,8000, 9000, 10000, 11000])
        screen = st.number_input('Screen Size (inches)', 6.0, 12.0, 6.7)

        if st.button('ทำนายราคาซื้อมือถือรุ่นใหม่'):
            result = predict_mobile_price_xgb(company_name, ram, front_cam, back_cam, processor, battery, screen)
            if result is not None:
                st.success(f'ราคาซื้อคาดการณ์: {result:.2f} ดอลลาร์')
            else:
                st.error("เกิดข้อผิดพลาดในการทำนายราคา")

    elif choice == 'Predict_Old_Mobile_Price(2000-2010)':
        st.subheader('ประเมินราคา ซื่อ ราคามือถือรุ่นเก่า')
        st.write(""" 
        ในส่วนนี้ เราจะใช้โมเดล Neural Network เพื่อทำนายราคามือถือรุ่นเก่าที่ออกระหว่างปี 2000 ถึง 2010 โดยใช้ข้อมูลต่างๆ เช่น ความจุแบตเตอรี่, ความเร็วของโปรเซสเซอร์, หน้าจอสัมผัส ฯลฯ
        ท่านสามารถกรอกข้อมูลมือถือที่ต้องการทำนายราคาได้จากฟอร์มด้านล่าง
        """)

        # รับอินพุตทั้งหมดที่โมเดลต้องการ (20 ฟีเจอร์)
        battery_power = st.selectbox('Battery Power (mAh)', [1000, 2000, 3000, 4000, 5000, 6000])
        blue = st.selectbox('Bluetooth Enabled (1 = Yes, 0 = No)', [0, 1])
        clock_speed = st.selectbox('Clock Speed (GHz)', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        dual_sim = st.selectbox('Dual SIM (1 = Yes, 0 = No)', [0, 1])
        fc = st.selectbox('Front Camera (MP)', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 16, 20])
        four_g = st.selectbox('4G Supported (1 = Yes, 0 = No)', [0, 1])
        int_memory = st.selectbox('Internal Memory (GB)', [4, 8, 16, 32, 64])
        storage = int_memory  # กำหนด storage ให้เป็นค่าเดียวกับ int_memory
        m_dep = st.selectbox('Mobile Depth (cm)', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        mobile_wt = st.selectbox('Mobile Weight (grams)', [100, 150, 200, 250])
        n_cores = st.selectbox('Number of CPU Cores', [1, 2, 4, 6, 8])
        pc = st.selectbox('Primary Camera (MP)', [5, 8, 10, 12, 16, 20, 50])
        px_height = st.selectbox('Pixel Height', [1000, 1500, 2000, 2500, 3000])
        px_width = st.selectbox('Pixel Width', [1000, 1500, 2000, 2500, 3000])
        ram = st.selectbox('RAM (GB)', [1, 2, 4, 8, 16])
        sc_h = st.selectbox('Screen Height (cm)', [5, 6, 7, 8, 9, 10, 12, 14])
        sc_w = st.selectbox('Screen Width (cm)', [5, 6, 7, 8, 9, 10, 12])
        talk_time = st.selectbox('Talk Time (hours)', [5, 10, 15, 20, 25, 30])
        three_g = st.selectbox('3G Supported (1 = Yes, 0 = No)', [0, 1])
        touch_screen = st.selectbox('Touch Screen (1 = Yes, 0 = No)', [0, 1])
        wifi = st.selectbox('Wi-Fi (1 = Yes, 0 = No)', [0, 1])

        if st.button('ทำนาย'):
            # ทำนายราคามือถือ
            result = predict_mobile_price_nn(battery_power, blue, clock_speed, dual_sim, fc, 
                                            four_g, int_memory, m_dep, mobile_wt, n_cores, pc, 
                                            px_height, px_width, ram, sc_h, sc_w, talk_time, 
                                            three_g, touch_screen, wifi)
            st.success(f'ราคาคาดการณ์: {result:.2f} บาท')



    elif choice == 'การฝึกฝนโมเดล Machine Learning':
        st.subheader("การฝึกฝนโมเดล Machine Learning (ML)")
        
        st.write("""
             ในส่วนนี้เราจะทำการฝึกฝนโมเดล Machine Learning สำหรับการทำนายราคามือถือจากข้อมูลที่มีอยู่
        เราใช้ XGBoost เป็นโมเดลที่ใช้ในการฝึกสอน และเราจะทำการประมวลผลข้อมูลเพื่อเตรียมข้อมูลให้พร้อมสำหรับการฝึกโมเดล
        """)
        st.write("""  เนื่องจาก XGBoost เป็นโมเดลหลักเพียงตัวเดียว ซึ่งมีความสามารถในการทำงานทั้งในกรณีของการทำนายค่าเชิงเส้น (regression) และการจัดประเภท (classification) 
        """)

        # การนำเข้าคลังข้อมูล (Imports)
        st.write("### การนำเข้าคลังข้อมูล (Imports):")
        st.code("""
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        import re
        from sklearn.preprocessing import LabelEncoder
        import joblib
        """)
        
        st.write("ในขั้นตอนนี้เราได้นำเข้าคลังข้อมูลที่จำเป็น เช่น `xgboost`, `sklearn`, `pandas`, `numpy`, `re`, `joblib` สำหรับการสร้างโมเดล, การประมวลผลข้อมูล, และการบันทึกโมเดลที่ฝึกแล้ว.")

        # การอ่านข้อมูลจากไฟล์ CSV
        st.write("### การอ่านข้อมูลจากไฟล์ CSV:")
        st.code("""
        apple_df = pd.read_csv("apple_mobile_price.csv", encoding='latin1')
        non_apple_df = pd.read_csv("non_apple_mobile_price.csv", encoding='latin1')
        """)
        
        st.write("เราจะอ่านข้อมูลจากไฟล์ CSV ที่มีข้อมูลเกี่ยวกับมือถือของ Apple และ Non-Apple โดยใช้คำสั่ง `pd.read_csv()`")

        # การกรองข้อมูลเฉพาะปี 2020-2025
        st.write("### การกรองข้อมูลเฉพาะปี 2020-2025:")
        st.code("""
        apple_df = apple_df[apple_df["Launched Year"].astype(str).str[:4].astype(float).between(2020, 2025)].copy()
        non_apple_df = non_apple_df[non_apple_df["Launched Year"].astype(str).str[:4].astype(float).between(2020, 2025)].copy()
        """)
        
        st.write("กรองข้อมูลให้เหลือเฉพาะมือถือที่เปิดตัวในปี 2020-2025 โดยใช้ฟังก์ชัน `astype()` และ `between()`")

        # การเติมค่า NaN ด้วยค่าเฉลี่ย
        st.write("### การเติมค่า NaN ด้วยค่าเฉลี่ย:")
        st.code("""
        apple_df.fillna(apple_df.mean(numeric_only=True), inplace=True)
        non_apple_df.fillna(non_apple_df.mean(numeric_only=True), inplace=True)
        """)
        
        st.write("ในขั้นตอนนี้เราจะเติมค่า NaN ในคอลัมน์ที่เป็นตัวเลขด้วยค่าเฉลี่ยของคอลัมน์นั้นๆ")

        # การทำความสะอาดข้อมูล
        st.write("### การทำความสะอาดข้อมูล:")
        st.code("""
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

        for df in [apple_df, non_apple_df]:
            df["Battery Capacity"] = df["Battery Capacity"].apply(clean_battery)
            df["RAM"] = df["RAM"].apply(clean_ram)
            df["Screen Size"] = df["Screen Size"].apply(clean_screen_size)
            df["Front Camera"] = df["Front Camera"].apply(clean_camera)
            df["Back Camera"] = df["Back Camera"].apply(clean_camera)
        """)
        
        st.write("ใช้ฟังก์ชันที่เราสร้างขึ้นเพื่อทำความสะอาดข้อมูล เช่น การแปลงค่าต่างๆ ในคอลัมน์แบตเตอรี่, RAM, ขนาดหน้าจอ และกล้องให้เป็นตัวเลข")

        # การแปลงข้อมูลหมวดหมู่เป็นตัวเลข
        st.write("### การแปลงข้อมูลหมวดหมู่เป็นตัวเลข:")
        st.code("""
        def encode_features(df):
            label_encoders = {}
            for col in ["Processor", "Company Name"]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            return df, label_encoders

        apple_df, apple_encoders = encode_features(apple_df)
        non_apple_df, non_apple_encoders = encode_features(non_apple_df)
        joblib.dump(apple_encoders, "apple_label_encoders.pkl")
        joblib.dump(non_apple_encoders, "non_apple_label_encoders.pkl")
        """)
        
        st.write("ใช้ `LabelEncoder` เพื่อแปลงข้อมูลหมวดหมู่ (เช่น `Processor` และ `Company Name`) ให้เป็นค่าตัวเลข")

        # การแยกข้อมูลและการแบ่งชุดข้อมูล (Train-Test Split)
        st.write("### การแยกข้อมูลและการแบ่งชุดข้อมูล (Train-Test Split):")
        st.code("""
        X_apple, y_apple = apple_df[features], apple_df["Launched Price (USA)"]
        X_non_apple, y_non_apple = non_apple_df[features], non_apple_df["Launched Price (USA)"]

        X_train_apple, X_test_apple, y_train_apple, y_test_apple = train_test_split(X_apple, y_apple, test_size=0.2, random_state=42)
        X_train_non_apple, X_test_non_apple, y_train_non_apple, y_test_non_apple = train_test_split(X_non_apple, y_non_apple, test_size=0.2, random_state=42)
        """)
        
        st.write("แบ่งข้อมูลออกเป็นฟีเจอร์และตัวแปรเป้าหมาย จากนั้นแบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ (80% ฝึกสอน, 20% ทดสอบ)")

        # การสร้างโมเดล XGBoost
        st.write("### การสร้างโมเดล XGBoost:")
        st.code("""
        apple_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=5, random_state=42)
        non_apple_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.08, max_depth=5, random_state=42)
        """)
        
        st.write("สร้างโมเดล XGBoost โดยกำหนดค่าพารามิเตอร์ต่างๆ เช่น จำนวนต้นไม้ (n_estimators), อัตราการเรียนรู้ (learning_rate), ความลึกสูงสุดของต้นไม้ (max_depth)")

        # การฝึกสอนโมเดล
        st.write("### การฝึกสอนโมเดล:")
        st.code("""
        apple_model.fit(X_train_apple, y_train_apple)
        non_apple_model.fit(X_train_non_apple, y_train_non_apple)
        """)
        
        st.write("ใช้ฟังก์ชัน `fit()` เพื่อฝึกสอนโมเดลด้วยข้อมูลชุดฝึกสอน")

        # การบันทึกโมเดลและ encoder
        st.write("### การบันทึกโมเดลและ encoder:")
        st.code("""
        joblib.dump(apple_model, "apple_xgboost_model.pkl")
        joblib.dump(non_apple_model, "non_apple_xgboost_model.pkl")
        joblib.dump(apple_encoders, "apple_label_encoders.pkl")
        joblib.dump(non_apple_encoders, "non_apple_label_encoders.pkl")
        """)
        
        st.write("บันทึกโมเดลและ encoder ที่ฝึกแล้วโดยใช้ `joblib.dump()`")

        # การทำนายราคา
        st.write("### การทำนายราคา:")
        st.code("""
        def predict_price(sample_data):
            df = pd.DataFrame(sample_data)
            company = df.iloc[0]['Company Name']
            if company == 'Apple':
                model = apple_model
                encoders = apple_encoders
            else:
                model = non_apple_model
                encoders = non_apple_encoders
            df["Processor"] = df["Processor"].apply(lambda x: encoders["Processor"].transform([x])[0] if x in encoders["Processor"].classes_ else 0)
            df["Company Name"] = encoders["Company Name"].transform(df["Company Name"])
            X_sample = df[features]
            return model.predict(X_sample)

        sample_phones = [{'RAM': 6, 'Battery Capacity': 4000, 'Screen Size': 6.7, 'Front Camera': 20, 'Back Camera': 42, 'Company Name': 'Apple', 'Processor': 'Apple A17'}, {'RAM': 8, 'Battery Capacity': 5000, 'Screen Size': 6.7, 'Front Camera': 20, 'Back Camera': 64, 'Company Name': 'Samsung', 'Processor': 'Snapdragon'}]

        for i, phone in enumerate(sample_phones):
            print(f"Prediction for phone {i+1}: ${predict_price([phone])[0]:.2f}")
        """)
        
        st.write("ฟังก์ชันทำนายราคามือถือโดยใช้ข้อมูลตัวอย่าง (Apple หรือ Non-Apple) และโมเดลที่เหมาะสม")


    elif choice == 'การฝึกฝนโมเดล  Neural Network':
        st.subheader("การฝึกฝนโมเดล  Neural Network เพื่อเปรียบเทียบกับอีก Machine Learning (หลายๆ โมเดล)")
        
        st.write("""
        ในส่วนนี้เราจะฝึกโมเดลหลายๆ แบบเพื่อทำนายราคามือถือจากข้อมูลที่มีอยู่ โดยจะใช้หลายโมเดลเช่น Linear Regression, KNN, Logistic Regression, Decision Tree, Random Forest และ Neural Network
        และเราจะประเมินประสิทธิภาพของแต่ละโมเดลในชุดข้อมูลทดสอบ
        """)

        # การนำเข้าคลังข้อมูล (Imports)
        st.write("### การนำเข้าคลังข้อมูล (Imports):")
        st.code("""
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from sklearn.preprocessing import StandardScaler
        """)
        
        st.write("ในขั้นตอนนี้เราได้นำเข้าคลังข้อมูลที่จำเป็นเช่น `pandas`, `numpy`, `sklearn`, `tensorflow` และ `joblib` เพื่อใช้ในการฝึกโมเดลและประเมินผล")

        # การโหลดข้อมูลจากไฟล์ CSV
        st.write("### การโหลดข้อมูลจากไฟล์ CSV:")
        st.code("""
        dataset = pd.read_csv('C:/Users/kittisak/OneDrive/Documents/AI/web/train.csv')
        """)
        
        st.write("เราจะโหลดชุดข้อมูลที่มีข้อมูลมือถือพร้อมราคาจากไฟล์ CSV")

        # การเตรียมข้อมูล
        st.write("### การเตรียมข้อมูล:")
        st.code("""
        X = dataset.drop('price_range', axis=1)
        y = dataset['price_range']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        """)
        
        st.write("เราแยกข้อมูลเป็นฟีเจอร์ (X) และตัวแปรเป้าหมาย (y) จากนั้นแบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ และทำการปรับขนาดข้อมูลให้เป็นมาตรฐานด้วย `StandardScaler`")

        # การฝึกโมเดล Linear Regression
        st.write("### การฝึกโมเดล Linear Regression:")
        st.code("""
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        lm_score = lm.score(X_test, y_test)
        """)
        
        st.write("ฝึกโมเดล Linear Regression โดยใช้ชุดข้อมูลฝึกสอน จากนั้นประเมินโมเดลด้วยข้อมูลทดสอบ")

        # การฝึกโมเดล KNN
        st.write("### การฝึกโมเดล KNN:")
        st.code("""
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)
        knn_score = knn.score(X_test, y_test)
        """)
        
        st.write("ฝึกโมเดล KNN (k-Nearest Neighbors) โดยใช้ข้อมูลชุดฝึกสอน")

        # การหาค่า K ที่เหมาะสมโดยใช้ Elbow Method
        st.write("### การหาค่า K ที่เหมาะสม (Elbow Method):")
        st.code("""
        error_rate = []
        for i in range(1, 20):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        """)
        
        st.write("ใช้ Elbow Method เพื่อตรวจสอบค่าของ K ที่เหมาะสมในการใช้กับโมเดล KNN")

        # การฝึกโมเดล Logistic Regression
        st.write("### การฝึกโมเดล Logistic Regression:")
        st.code("""
        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)
        logmodel_score = logmodel.score(X_test, y_test)
        """)
        
        st.write("ฝึกโมเดล Logistic Regression และประเมินผล")

        # การฝึกโมเดล Decision Tree
        st.write("### การฝึกโมเดล Decision Tree:")
        st.code("""
        dtree = DecisionTreeClassifier()
        dtree.fit(X_train, y_train)
        dtree_score = dtree.score(X_test, y_test)
        """)
        
        st.write("ฝึกโมเดล Decision Tree และประเมินผล")

        # การฝึกโมเดล Random Forest
        st.write("### การฝึกโมเดล Random Forest:")
        st.code("""
        rfc = RandomForestClassifier(n_estimators=200)
        rfc.fit(X_train, y_train)
        rfc_score = rfc.score(X_test, y_test)
        """)
        
        st.write("ฝึกโมเดล Random Forest และประเมินผล")

        # การสร้างและฝึก Neural Network
        st.write("### การสร้างและฝึก Neural Network:")
        st.code("""
        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
        test_loss = model.evaluate(X_test, y_test)
        model.save('mobile_price_model.keras')
        """)
        
        st.write("สร้าง Neural Network โดยมีชั้นซ่อน 3 ชั้น และใช้ Dropout เพื่อหลีกเลี่ยงการ Overfitting")

        # การแสดงผลการฝึกโมเดล
        st.write("### การแสดงผลการฝึกโมเดล:")
        st.code("""
        print(f"Linear Regression Score: {lm_score}")
        print(f"KNN Score: {knn_score}")
        print(f"Logistic Regression Score: {logmodel_score}")
        print(f"Decision Tree Score: {dtree_score}")
        print(f"Random Forest Score: {rfc_score}")
        """)
        
        st.write("แสดงคะแนนจากโมเดลที่ฝึกมาแล้ว")

        # การใช้ KNN สำหรับทำนายราคา
        st.write("### การใช้ KNN สำหรับทำนายราคา:")
        st.code("""
        knn_pred = knn.predict(X_test)
        print(classification_report(y_test, knn_pred))
        cm = confusion_matrix(y_test, knn_pred)
        print(cm)
        """)
        
        st.write("ใช้โมเดล KNN เพื่อทำนายราคาและแสดงผลการประเมินผล เช่น classification report และ confusion matrix")

        # การทำนายราคาในชุดข้อมูลทดสอบ
        st.write("### การทำนายราคาในชุดข้อมูลทดสอบ:")
        st.code("""
        data_test = pd.read_csv('C:/Users/kittisak/OneDrive/Documents/AI/web/test.csv')
        data_test = data_test.drop('id', axis=1)
        predicted_price = knn.predict(data_test)
        data_test['price_range'] = predicted_price
        data_test.to_csv('predicted_test_results.csv', index=False)
        joblib.dump(knn, 'knn_model.pkl')
        """)
        
        st.write("ทำนายราคาของมือถือในชุดข้อมูลทดสอบและบันทึกผลลัพธ์ลงในไฟล์ CSV พร้อมทั้งบันทึกโมเดล KNN สำหรับใช้งานในอนาคต")
        
    elif choice == 'About':
        st.subheader('📌 About This Project')
        st.write("""
            - This project is developed using **Streamlit** for the frontend interface.
            - It utilizes **Machine Learning** models such as **Linear Regression**, **K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Decision Trees**, **Random Forest**, and **Neural Networks** for mobile price prediction.
            - The project is deployed using **Streamlit**, allowing easy interaction and real-time predictions.
        """)

        st.subheader('👨‍💻 Developer Information')
        st.write("""
            - **Creator**: **Kittisak Tantrtone** (6604062610021)
            - **Email**: [s6604062610021@email.kmutnb.ac.th](mailto:s6604062610021@email.kmutnb.ac.th)
            - **GitHub**: [A-Phrom-A](https://github.com/A-Phrom-A)
        """)

        st.subheader('📊 Datasets Used')
        st.write("""
            - **Predict Old Mobile Price (2000-2010)**  
            [Dataset Link](https://www.kaggle.com/code/melissamonfared/mobile-price-prediction-eda-classification?select=test.csv)
            - **New Mobile Price Prediction (2020-2025)**  
            [Dataset Link](https://www.kaggle.com/code/hanymato/mobile-price-prediction-model/input)
        """)

        st.write("💡 *Feel free to reach out for inquiries, collaborations, or further details about this project!* 🚀")



if __name__ == '__main__':
    main()
