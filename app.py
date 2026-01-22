import streamlit as st
import numpy as np
import joblib

# ---------------------------------
# Load trained models and scaler
# ---------------------------------
scaler = joblib.load("scaler.pkl")

svm_linear = joblib.load("svm_linear.pkl")
svm_poly = joblib.load("svm_poly.pkl")
svm_rbf = joblib.load("loan_svm_model.pkl")   # Best model (RBF)

# ---------------------------------
# App Title & Description
# ---------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("💳 Smart Loan Approval System")

st.write(
    """
    **This system uses Support Vector Machines (SVM) to predict loan approval.**  
    Different SVM kernels are used to understand linear and non-linear decision boundaries
    in fintech loan approval scenarios.
    """
)

# ---------------------------------
# Input Section
# ---------------------------------
st.header("📥 Enter Applicant Details")

applicant_income = st.number_input(
    "Applicant Income", min_value=0, value=5000
)

loan_amount = st.number_input(
    "Loan Amount", min_value=0, value=150
)

credit_history = st.selectbox(
    "Credit History",
    ["Yes", "No"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self Employed"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Encoding inputs
credit_history = 1 if credit_history == "Yes" else 0
employment_status = 0 if employment_status == "Salaried" else 1
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ---------------------------------
# Model Selection
# ---------------------------------
st.header("🧠 Select SVM Kernel")

kernel = st.radio(
    "Choose SVM Model",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel == "Linear SVM":
    model = svm_linear
elif kernel == "Polynomial SVM":
    model = svm_poly
else:
    model = svm_rbf

# ---------------------------------
# Prediction Button
# ---------------------------------
if st.button("🔍 Check Loan Eligibility"):

    input_data = np.array([
        applicant_income,
        0,  # Coapplicant income (kept 0)
        loan_amount,
        360,  # Loan term (default)
        credit_history
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    confidence = model.decision_function(input_scaled)[0]

    # ---------------------------------
    # Output Section
    # ---------------------------------
    st.header("📊 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
        explanation = (
            "Based on credit history and income pattern, "
            "the applicant is **likely to repay the loan**."
        )
    else:
        st.error("❌ Loan Rejected")
        explanation = (
            "Based on credit history and income pattern, "
            "the applicant is **unlikely to repay the loan**."
        )

    st.write("**Kernel Used:**", kernel)
    st.write("**Confidence Score:**", round(confidence, 2))

    # ---------------------------------
    # Business Explanation
    # ---------------------------------
    st.subheader("📌 Business Explanation")
    st.info(explanation)
