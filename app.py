import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="IS Project: Data Science", layout="wide", page_icon="🚀")

@st.cache_resource
def load_all_models():
    models = {}
    try:
        # โหลดโมเดล Loan จากโฟลเดอร์ models/
        models['loan_scaler'] = joblib.load('models/scaler.pkl')
        models['loan_ensemble'] = joblib.load('models/ensemble_model.pkl')
        models['loan_nn'] = load_model('models/nn_model.h5')
        
        # โหลดโมเดล Titanic จากโฟลเดอร์ models/
        models['titanic_scaler'] = joblib.load('models/scaler_titanic.pkl')
        models['titanic_ensemble'] = joblib.load('models/ensemble_titanic.pkl')
        models['titanic_nn'] = load_model('models/nn_titanic.h5')
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลดไฟล์โมเดล: {e}")
        st.warning("กรุณาตรวจสอบว่าคุณได้รันไฟล์ train_models.py แล้ว และมีไฟล์ .pkl, .h5 อยู่ในโฟลเดอร์ 'models/'")
    return models

models = load_all_models()
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8637/8637123.png", width=100) 
    st.title("เมนูหลัก")
    
    page = st.radio(
        "📄 เลือกหน้าเว็บ:",
        ["🏠 หน้าหลักโปรเจค (Home)", 
         "📚 1. ทฤษฎี: Ensemble Model", 
         "⚙️ 2. ทดสอบ: Ensemble Model", 
         "🧠 3. ทฤษฎี: Neural Network", 
         "⚡ 4. ทดสอบ: Neural Network"]
    )
    
    st.markdown("---")
    
    if page != "🏠 หน้าหลักโปรเจค (Home)":
        dataset_choice = st.selectbox(
            "📂 เลือกชุดข้อมูลเพื่อทดสอบ:",
            ["1. Loan Approval (พิจารณาสินเชื่อ)", "2. Titanic Survival (พยากรณ์ผู้รอดชีวิต)"]
        )
    else:
        dataset_choice = None 
    
    st.markdown("---")
    st.caption("💡 **Project IS 2568**\n\nพัฒนาโดย: [ณภัทร เพชรทอง]")

if page == "🏠 หน้าหลักโปรเจค (Home)":
    # ส่วน Header Banner
    st.markdown("""
    <div style="background-color:#0E1117; padding:20px; border-radius:10px; border: 1px solid #4B4B4B; text-align:center;">
        <h1 style="color:#4A90E2;">🚀 Project IS: Machine Learning & Deep Learning</h1>
        <p style="color:#A0AEC0; font-size:18px;">ระบบพยากรณ์อัจฉริยะที่พัฒนาขึ้นเพื่อการศึกษา กระบวนการตั้งแต่ Data Preprocessing สู่การสร้าง Web Application</p>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    st.write("### 📌 ภาพรวมของชุดข้อมูลที่ใช้พัฒนา (Datasets)")
    
    # จัด Layout แบบคอลัมน์คู่ (Cards)
    col1, col2 = st.columns(2)
    with col1:
        st.info("#### 🏦 1. Loan Approval Prediction")
        st.write("**คำอธิบาย:** ชุดข้อมูลจำลองกระบวนการพิจารณาสินเชื่อธนาคาร โดยวิเคราะห์จากข้อมูลส่วนบุคคลและข้อมูลทางการเงิน")
        st.write("📊 **จำนวนข้อมูล:** 1,000 แถว")
        st.write("🎯 **เป้าหมาย:** ทำนายการ `อนุมัติ` หรือ `ไม่อนุมัติ` สินเชื่อ")
        
    with col2:
        st.success("#### 🚢 2. Titanic Survival Prediction")
        st.write("**คำอธิบาย:** ชุดข้อมูลประวัติศาสตร์เพื่อทำนายอัตราการรอดชีวิตจากโศกนาฏกรรมเรือไททานิค")
        st.write("📊 **จำนวนข้อมูล:** 891 แถว")
        st.write("🎯 **เป้าหมาย:** ทำนายว่าผู้โดยสาร `รอดชีวิต` หรือ `เสียชีวิต`")

    st.markdown("---")
    
    # เพิ่มส่วนแหล่งอ้างอิงข้อมูล (Data Sources)
    st.write("### 📚 แหล่งอ้างอิงข้อมูล (Data Sources & References)")
    
    # ใช้ Expander หรือจัดเป็น Markdown ให้อ่านง่าย
    st.markdown("""
    **1. ชุดข้อมูล Loan Approval Prediction (พิจารณาสินเชื่อ)**
    * **แหล่งที่มา:** สร้างขึ้นจำลอง (Generated Dataset) ด้วยไลบรารี `numpy` และ `pandas` ในภาษา Python
    * **เหตุผลที่ใช้:** เพื่อจำลองสถานการณ์ข้อมูลโลกจริงที่มีความไม่สมบูรณ์ (Dirty Data) โดยมีการตั้งใจใส่ค่าว่าง (Missing Values) และค่าผิดปกติ (Outliers) เข้าไป เพื่อให้เป็นไปตามข้อกำหนดในการฝึกฝนขั้นตอน Data Preprocessing และ Data Cleaning อย่างครบถ้วน

    **2. ชุดข้อมูล Titanic Survival Prediction (พยากรณ์ผู้รอดชีวิต)**
    * **แหล่งที่มา:** ดาวน์โหลดจากเว็บไซต์ Kaggle ในการแข่งขัน "Titanic: Machine Learning from Disaster" 
    * **ลิงก์อ้างอิง:** [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
    * **เหตุผลที่ใช้:** เป็นชุดข้อมูลคลาสสิกที่มีความไม่สมบูรณ์ในบางคอลัมน์ (เช่น อายุ และท่าเรือ) เหมาะสำหรับการนำมาทำ Data Cleaning และทดสอบประสิทธิภาพของ Machine Learning และ Neural Network
    """)
    
    st.markdown("---")
    
    # ส่วนสถิติและเทคโนโลยี
    st.write("### ⚙️ สถาปัตยกรรมโมเดลที่ใช้งาน")
    col3, col4, col5 = st.columns(3)
    col3.metric("Machine Learning", "Ensemble", "Soft Voting")
    col4.metric("Deep Learning", "Neural Network", "2 Hidden Layers")
    col5.metric("Web Framework", "Streamlit", "Python")

elif dataset_choice == "1. Loan Approval (พิจารณาสินเชื่อ)":
    if page == "📚 1. ทฤษฎี: Ensemble Model":
        st.title("🏦 ทฤษฎี: Ensemble Model (Loan Approval)")
        st.write("---")
        st.subheader("1. แหล่งที่มาและการเตรียมข้อมูล (Data Preprocessing)")
        st.write("- **การทำความสะอาดข้อมูล (Data Cleaning):** เติมค่า Median ในคอลัมน์อายุ, รายได้, และคะแนนเครดิตที่หายไป จัดการรายได้ที่ติดลบ และแปลงเพศเป็น 0 กับ 1 พร้อมปรับสเกลข้อมูล")
        st.subheader("2. อัลกอริทึมที่ใช้")
        st.write("ใช้ **Soft Voting Classifier** จาก 3 โมเดล: Random Forest, Gradient Boosting และ Logistic Regression")

    elif page == "⚙️ 2. ทดสอบ: Ensemble Model":
        st.title("🏦 ทดสอบทำนายผล: Ensemble Model (Loan)")
        with st.form("loan_ensemble_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.radio("เพศ", ["ชาย", "หญิง"])
                age = st.number_input("อายุ", min_value=18, max_value=100, value=30)
                income = st.number_input("รายได้ (บาท/เดือน)", min_value=0, value=50000, step=1000)
            with col2:
                loan_amount = st.number_input("วงเงินกู้ที่ต้องการ", min_value=0, value=150000, step=10000)
                credit_score = st.number_input("คะแนนเครดิต (300-850)", min_value=300, max_value=850, value=650)
            submit = st.form_submit_button("🔍 วิเคราะห์สินเชื่อ")
            
        if submit:
            gen_val = 1 if gender == "ชาย" else 0
            input_data = np.array([[age, gen_val, income, loan_amount, credit_score]])
            input_scaled = models['loan_scaler'].transform(input_data)
            pred = models['loan_ensemble'].predict(input_scaled)
            
            if pred[0] == 1:
                st.success("🎉 ผลการวิเคราะห์: **อนุมัติสินเชื่อ**")
            else:
                st.error("❌ ผลการวิเคราะห์: **ไม่อนุมัติสินเชื่อ**")

    elif page == "🧠 3. ทฤษฎี: Neural Network":
        st.title("🏦 ทฤษฎี: Neural Network (Loan Approval)")
        st.write("---")
        st.write("**โครงสร้างสถาปัตยกรรม (Model Architecture):**")
        st.write("- **Input Layer:** รับข้อมูล 5 Features ที่ผ่านการทำ Scale แล้ว")
        st.write("- **Hidden Layers:** ใช้ 2 ชั้น (32 และ 16 โหนด) ทำงานร่วมกับ ReLU Activation")
        st.write("- **Output Layer:** ใช้ 1 โหนด พร้อม Sigmoid Activation สำหรับทำนายผลแบบ Binary")

    elif page == "⚡ 4. ทดสอบ: Neural Network":
        st.title("🏦 ทดสอบทำนายผล: Neural Network (Loan)")
        with st.form("loan_nn_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.radio("เพศ", ["ชาย", "หญิง"])
                age = st.number_input("อายุ", min_value=18, max_value=100, value=30)
                income = st.number_input("รายได้ (บาท/เดือน)", min_value=0, value=50000, step=1000)
            with col2:
                loan_amount = st.number_input("วงเงินกู้ที่ต้องการ", min_value=0, value=150000, step=10000)
                credit_score = st.number_input("คะแนนเครดิต (300-850)", min_value=300, max_value=850, value=650)
            submit = st.form_submit_button("⚡ ทำนายด้วย Neural Network")
            
        if submit:
            gen_val = 1 if gender == "ชาย" else 0
            input_data = np.array([[age, gen_val, income, loan_amount, credit_score]])
            input_scaled = models['loan_scaler'].transform(input_data)
            pred_prob = models['loan_nn'].predict(input_scaled)
            pred = 1 if pred_prob[0][0] > 0.5 else 0
            
            if pred == 1:
                st.success(f"🎉 ผลการวิเคราะห์: **อนุมัติสินเชื่อ** (ความมั่นใจ: {pred_prob[0][0]*100:.2f}%)")
            else:
                st.error(f"❌ ผลการวิเคราะห์: **ไม่อนุมัติสินเชื่อ** (ความมั่นใจว่าผ่าน: {pred_prob[0][0]*100:.2f}%)")

elif dataset_choice == "2. Titanic Survival (พยากรณ์ผู้รอดชีวิต)":
    if page == "📚 1. ทฤษฎี: Ensemble Model":
        st.title("🚢 ทฤษฎี: Ensemble Model (Titanic)")
        st.write("---")
        st.subheader("1. แหล่งที่มาและการเตรียมข้อมูล (Data Preprocessing)")
        st.write("- **การทำความสะอาดข้อมูล:** เติมค่า Median ในคอลัมน์ Age, เติม Mode ใน Embarked และตัดคอลัมน์ที่ไม่จำเป็นทิ้ง แปลงข้อมูลตัวอักษรเป็นตัวเลข")
        st.subheader("2. อัลกอริทึมที่ใช้")
        st.write("ใช้ **Soft Voting Classifier** รวม 3 โมเดล: Random Forest, Gradient Boosting และ Logistic Regression")

    elif page == "⚙️ 2. ทดสอบ: Ensemble Model":
        st.title("🚢 ทดสอบทำนายผล: Ensemble Model (Titanic)")
        with st.form("titanic_ensemble_form"):
            col1, col2 = st.columns(2)
            with col1:
                pclass = st.selectbox("ระดับชั้นผู้โดยสาร (Pclass)", [1, 2, 3])
                sex = st.radio("เพศ", ["ชาย", "หญิง"])
                age = st.number_input("อายุ", min_value=0, max_value=100, value=30)
                fare = st.number_input("ค่าตั๋ว (Fare)", min_value=0.0, value=32.0)
            with col2:
                sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสบนเรือ (SibSp)", min_value=0, max_value=10, value=0)
                parch = st.number_input("จำนวนพ่อแม่/ลูกบนเรือ (Parch)", min_value=0, max_value=10, value=0)
                embarked = st.selectbox("ท่าเรือที่ขึ้น (Embarked)", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            submit = st.form_submit_button("🔍 ทำนายโอกาสรอดชีวิต")
            
        if submit:
            sex_val = 1 if sex == "ชาย" else 0
            embarked_val = 0 if "S" in embarked else (1 if "C" in embarked else 2)
            input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
            input_scaled = models['titanic_scaler'].transform(input_data)
            pred = models['titanic_ensemble'].predict(input_scaled)
            
            if pred[0] == 1:
                st.success("🎉 ผลทำนาย: **รอดชีวิต (Survived)**")
            else:
                st.error("💀 ผลทำนาย: **เสียชีวิต (Not Survived)**")

    elif page == "🧠 3. ทฤษฎี: Neural Network":
        st.title("🚢 ทฤษฎี: Neural Network (Titanic)")
        st.write("---")
        st.write("**โครงสร้างสถาปัตยกรรม (Model Architecture):**")
        st.write("- **Input Layer:** รับค่า 7 Features")
        st.write("- **Hidden Layers:** 2 ชั้น (16 โหนด และ 8 โหนด) พร้อม Activation Function แบบ ReLU")
        st.write("- **Output Layer:** ใช้ 1 โหนด พร้อม Sigmoid สำหรับทำนายค่าความน่าจะเป็น 0 ถึง 1")

    elif page == "⚡ 4. ทดสอบ: Neural Network":
        st.title("🚢 ทดสอบทำนายผล: Neural Network (Titanic)")
        with st.form("titanic_nn_form"):
            col1, col2 = st.columns(2)
            with col1:
                pclass = st.selectbox("ระดับชั้นผู้โดยสาร (Pclass)", [1, 2, 3])
                sex = st.radio("เพศ", ["ชาย", "หญิง"])
                age = st.number_input("อายุ", min_value=0, max_value=100, value=30)
                fare = st.number_input("ค่าตั๋ว (Fare)", min_value=0.0, value=32.0)
            with col2:
                sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสบนเรือ (SibSp)", min_value=0, max_value=10, value=0)
                parch = st.number_input("จำนวนพ่อแม่/ลูกบนเรือ (Parch)", min_value=0, max_value=10, value=0)
                embarked = st.selectbox("ท่าเรือที่ขึ้น (Embarked)", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            submit = st.form_submit_button("⚡ ทำนายด้วย Neural Network")
            
        if submit:
            sex_val = 1 if sex == "ชาย" else 0
            embarked_val = 0 if "S" in embarked else (1 if "C" in embarked else 2)
            input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
            input_scaled = models['titanic_scaler'].transform(input_data)
            pred_prob = models['titanic_nn'].predict(input_scaled)
            pred = 1 if pred_prob[0][0] > 0.5 else 0
            
            if pred == 1:
                st.success(f"🎉 ผลทำนาย: **รอดชีวิต** (ความมั่นใจ: {pred_prob[0][0]*100:.2f}%)")
            else:
                st.error(f"💀 ผลทำนาย: **เสียชีวิต** (ความมั่นใจว่ารอด: {pred_prob[0][0]*100:.2f}%)")