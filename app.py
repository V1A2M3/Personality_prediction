import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

st.title("🧠 Personality Prediction App")
st.write("Upload a training dataset to train the model, then upload test data to evaluate and predict.")

# Upload training dataset
train_file = st.file_uploader("📁 Upload Training CSV", type=["csv"])
test_file = st.file_uploader("📁 Upload Test CSV", type=["csv"])

if train_file is not None and test_file is not None:
    # Load training data
    data = pd.read_csv(train dataset)

    # Preprocess training data
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])

    input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
    output_col = 'Personality (Class label)'

    scaler = StandardScaler()
    data[input_cols] = scaler.fit_transform(data[input_cols])

    X_train = data[input_cols]
    y_train = data[output_col]

    # Train model
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # Load and preprocess test data
    test_data = pd.read_csv(test dataset)
    test_data['Gender'] = le.transform(test_data['Gender'])  # use same encoder
    test_data[input_cols] = scaler.transform(test_data[input_cols])

    X_test = test_data[input_cols]
    y_test = test_data['Personality (class label)']

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write("### 🎯 Prediction Results")
    st.write("**Accuracy:**", f"{acc * 100:.2f}%")
    test_data['Predicted Personality'] = y_pred
    st.dataframe(test_data)

    # Optional: Download predictions
    csv = test_data.to_csv(index=False)
    st.download_button("📥 Download Predictions", csv, "predicted_personality.csv", "text/csv")
else:
    st.info("⬆️ Please upload both training and test CSV files to begin.")
