import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

print("="*50)
print("🚀 เริ่มต้นกระบวนการ Train Models สำหรับ Project IS")
print("="*50)

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ==========================================
# 📊 DATASET 1: LOAN APPROVAL
# ==========================================
print("\n[1/2] กำลังประมวลผล Dataset 1: Loan Approval...")

# 1.1 สร้าง Dirty Dataset จำลอง
np.random.seed(42)
n_samples = 1000
data_loan = {
    'Age': np.random.randint(20, 70, n_samples).astype(float),
    'Gender': np.random.choice(['Male', 'Female', 'M', 'F', np.nan], n_samples, p=[0.4, 0.4, 0.05, 0.05, 0.1]),
    'Income': np.random.normal(50000, 20000, n_samples),
    'LoanAmount': np.random.normal(150000, 50000, n_samples),
    'CreditScore': np.random.randint(300, 850, n_samples).astype(float),
    'LoanApproved': np.random.choice([1, 0], n_samples, p=[0.6, 0.4])
}
data_loan['Age'][::15] = np.nan
data_loan['Income'][::20] = -5000
data_loan['CreditScore'][::30] = 9999

df_loan_dirty = pd.DataFrame(data_loan)
# เซฟไฟล์ไปที่โฟลเดอร์ data
df_loan_dirty.to_csv("data/dirty_loan_data.csv", index=False)
print("  - สร้างไฟล์ 'data/dirty_loan_data.csv' สำเร็จ!")

# 1.2 Data Preprocessing
df_loan = pd.read_csv("data/dirty_loan_data.csv")
df_loan['Gender'] = df_loan['Gender'].replace({'M': 'Male', 'F': 'Female'})
df_loan.loc[df_loan['Income'] < 0, 'Income'] = np.nan
df_loan.loc[df_loan['CreditScore'] > 850, 'CreditScore'] = np.nan

df_loan['Age'] = df_loan['Age'].fillna(df_loan['Age'].median())
df_loan['Income'] = df_loan['Income'].fillna(df_loan['Income'].median())
df_loan['CreditScore'] = df_loan['CreditScore'].fillna(df_loan['CreditScore'].median())
df_loan['Gender'] = df_loan['Gender'].fillna(df_loan['Gender'].mode()[0])
df_loan['Gender'] = df_loan['Gender'].map({'Male': 1, 'Female': 0})

X_loan = df_loan.drop('LoanApproved', axis=1)
y_loan = df_loan['LoanApproved']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

scaler_loan = StandardScaler()
X_train_l_scaled = scaler_loan.fit_transform(X_train_l)
X_test_l_scaled = scaler_loan.transform(X_test_l)
# เซฟโมเดลไปที่โฟลเดอร์ models
joblib.dump(scaler_loan, 'models/scaler.pkl')

print("  - กำลังเทรน Ensemble Model (Loan)...")
clf1_l = RandomForestClassifier(n_estimators=50, random_state=42)
clf2_l = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf3_l = LogisticRegression(random_state=42)
ensemble_loan = VotingClassifier(estimators=[('rf', clf1_l), ('gb', clf2_l), ('lr', clf3_l)], voting='soft')
ensemble_loan.fit(X_train_l_scaled, y_train_l)
joblib.dump(ensemble_loan, 'models/ensemble_model.pkl')

print("  - กำลังเทรน Neural Network (Loan)...")
nn_loan = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_l_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_loan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_loan.fit(X_train_l_scaled, y_train_l, epochs=30, batch_size=16, verbose=0)
nn_loan.save('models/nn_model.h5')
print("  ✅ สำเร็จ! บันทึกโมเดล Loan Approval ลงโฟลเดอร์ models/ เรียบร้อยแล้ว")


# ==========================================
# 🚢 DATASET 2: TITANIC SURVIVAL
# ==========================================
print("\n[2/2] กำลังประมวลผล Dataset 2: Titanic Survival...")

# ตรวจสอบไฟล์ในโฟลเดอร์ data
if not os.path.exists('data/titanic.csv'):
    print("  ❌ ERROR: ไม่พบไฟล์ 'titanic.csv' ในโฟลเดอร์ data/")
    print("  กรุณานำไฟล์ titanic.csv ไปใส่ไว้ในโฟลเดอร์ data ก่อนรันอีกครั้ง")
else:
    # อ่านไฟล์จากโฟลเดอร์ data
    df_titanic = pd.read_csv('data/titanic.csv')
    df_titanic = df_titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Age'].median())
    df_titanic['Embarked'] = df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0])
    
    df_titanic['Sex'] = df_titanic['Sex'].map({'male': 1, 'female': 0})
    df_titanic['Embarked'] = df_titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    X_titanic = df_titanic.drop('Survived', axis=1)
    y_titanic = df_titanic['Survived']
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)
    
    scaler_titanic = StandardScaler()
    X_train_t_scaled = scaler_titanic.fit_transform(X_train_t)
    X_test_t_scaled = scaler_titanic.transform(X_test_t)
    # เซฟโมเดลไปที่โฟลเดอร์ models
    joblib.dump(scaler_titanic, 'models/scaler_titanic.pkl')
    
    print("  - กำลังเทรน Ensemble Model (Titanic)...")
    clf1_t = RandomForestClassifier(n_estimators=50, random_state=42)
    clf2_t = GradientBoostingClassifier(n_estimators=50, random_state=42)
    clf3_t = LogisticRegression(random_state=42)
    ensemble_titanic = VotingClassifier(estimators=[('rf', clf1_t), ('gb', clf2_t), ('lr', clf3_t)], voting='soft')
    ensemble_titanic.fit(X_train_t_scaled, y_train_t)
    joblib.dump(ensemble_titanic, 'models/ensemble_titanic.pkl')
    
    print("  - กำลังเทรน Neural Network (Titanic)...")
    nn_titanic = Sequential([
        Dense(16, activation='relu', input_shape=(X_train_t_scaled.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_titanic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_titanic.fit(X_train_t_scaled, y_train_t, epochs=50, batch_size=32, verbose=0)
    nn_titanic.save('models/nn_titanic.h5')
    print("  ✅ สำเร็จ! บันทึกโมเดล Titanic Survival ลงโฟลเดอร์ models/ เรียบร้อยแล้ว")

print("\n🎉 กระบวนการทั้งหมดเสร็จสมบูรณ์! รัน Streamlit (app.py) ได้เลยครับ")
print("="*50)