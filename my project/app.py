import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("🎓 Engineering Student Placement Predictor")

@st.cache_data
def load_data():
    df = pd.read_csv("btech_data.csv")
    # Convert Hostel to Yes/No if needed
    df['Hostel'] = df['Hostel'].apply(lambda x: 'Yes' if str(x).lower() in ['yes', '1', 'true'] else 'No')
    # Convert HistoryOfBacklogs to Yes/No if needed
    df['HistoryOfBacklogs'] = df['HistoryOfBacklogs'].apply(lambda x: 'Yes' if str(x).lower() in ['yes', '1', 'true'] else 'No')
    # Convert PlacedOrNot to Yes/No if needed
    df['PlacedOrNot'] = df['PlacedOrNot'].apply(lambda x: 'Yes' if str(x).lower() in ['yes', '1', 'true'] else 'No')
    return df

df = load_data()
label_encoders = {}
df_clean = df.copy()

# Encode categorical columns
for col in ['Gender', 'Stream', 'Internships', 'Hostel', 'HistoryOfBacklogs', 'PlacedOrNot']:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Features and target
X = df_clean.drop('PlacedOrNot', axis=1)
y = df_clean['PlacedOrNot']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit inputs
st.subheader("📋 Enter Student Details")

age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 20)
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
stream = st.selectbox("Stream", label_encoders['Stream'].classes_)
internships = st.selectbox("Internships", label_encoders['Internships'].classes_)
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1)
hostel = st.selectbox("Hostel", ['Yes', 'No'])
backlogs = st.selectbox("History Of Backlogs", ['Yes', 'No'])

# Prepare input data for prediction
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': label_encoders['Gender'].transform([gender])[0],
    'Stream': label_encoders['Stream'].transform([stream])[0],
    'Internships': label_encoders['Internships'].transform([internships])[0],
    'CGPA': cgpa,
    'Hostel': label_encoders['Hostel'].transform([hostel])[0],
    'HistoryOfBacklogs': label_encoders['HistoryOfBacklogs'].transform([backlogs])[0]
}])

# Predict on button click
if st.button("Predict Placement"):
    prediction = model.predict(input_data)[0]
    placed_label = label_encoders['PlacedOrNot'].inverse_transform([prediction])[0]
    st.success(f"🎯 Prediction: {placed_label}")
