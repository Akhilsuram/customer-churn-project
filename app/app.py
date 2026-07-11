import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load Model
model = joblib.load("models/churn_model.pkl")

# Title
st.title("📊 AI-Powered Customer Churn Prediction System")

st.markdown("""
This application predicts whether a customer is likely to churn
based on subscription, billing, and service information.
""")

# Sidebar
st.sidebar.header("About Project")

st.sidebar.write("""
This project uses Machine Learning to predict customer churn.

Models Used:
- Logistic Regression
- Random Forest
- XGBoost

Built with:
- Python
- Scikit-learn
- Streamlit
""")

# User Inputs
st.subheader("Enter Customer Details")

tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Create Input DataFrame
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Manual One-Hot Encoding
input_data["Contract_One year"] = 1 if contract == "One year" else 0
input_data["Contract_Two year"] = 1 if contract == "Two year" else 0

input_data["InternetService_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
input_data["InternetService_No"] = 1 if internet_service == "No" else 0

input_data["PaymentMethod_Credit card (automatic)"] = 1 if payment_method == "Credit card (automatic)" else 0
input_data["PaymentMethod_Electronic check"] = 1 if payment_method == "Electronic check" else 0
input_data["PaymentMethod_Mailed check"] = 1 if payment_method == "Mailed check" else 0

# Match Training Features
model_features = model.get_booster().feature_names

for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder Columns
input_data = input_data[model_features]

# Prediction
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    # Probability Display
    st.progress(float(probability))

    st.metric(
        label="Churn Probability",
        value=f"{probability * 100:.2f}%"
    )

    # Risk Category
    if probability > 0.7:
        st.warning("🔴 High Risk Customer")

    elif probability > 0.4:
        st.info("🟠 Medium Risk Customer")

    else:
        st.success("🟢 Low Risk Customer")

# Footer
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")