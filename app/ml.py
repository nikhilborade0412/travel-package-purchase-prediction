import streamlit as st
import pickle
import pandas as pd

# --------------------------------------------------
# Page Config (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="Travel Package Purchase Prediction",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# Page Headings
# --------------------------------------------------
st.title("🌍 Travel Package Purchase Prediction")
st.subheader("Predict Purchase for New Customer")
st.divider()


# --------------------------------------------------
# Dark Theme Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model & Preprocessor
# --------------------------------------------------
with open("pkl/tourism_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("pkl/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# --------------------------------------------------
# INPUT GRID (3 COLUMNS PER ROW)
# --------------------------------------------------
user_input = {}

# -------- Row 1 --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["Age"] = st.number_input("Age", 18, 70, 36)
with c2:
    user_input["CityTier"] = st.number_input("CityTier", 1, 3, 1)
with c3:
    user_input["DurationOfPitch"] = st.number_input("DurationOfPitch", 1, 60, 20)

# -------- Row 2 --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["NumberOfPersonVisiting"] = st.number_input(
        "NumberOfPersonVisiting", 1, 10, 3
    )
with c2:
    user_input["NumberOfFollowups"] = st.number_input(
        "NumberOfFollowups", 0, 10, 4
    )
with c3:
    user_input["PreferredPropertyStar"] = st.number_input(
        "PreferredPropertyStar", 1, 5, 3
    )

# -------- Row 3 --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["NumberOfTrips"] = st.number_input(
        "NumberOfTrips", 0, 20, 3
    )
with c2:
    user_input["Passport"] = st.number_input("Passport", 0, 1, 1)
with c3:
    user_input["PitchSatisfactionScore"] = st.number_input(
        "PitchSatisfactionScore", 1, 5, 3
    )

# -------- Row 4 --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["OwnCar"] = st.number_input("OwnCar", 0, 1, 1)
with c2:
    user_input["NumberOfChildrenVisiting"] = st.number_input(
        "NumberOfChildrenVisiting", 0, 10, 0
    )
with c3:
    user_input["MonthlyIncome"] = st.number_input(
        "MonthlyIncome", 5000, 200000, 22418
    )

# -------- Row 5 (Categorical) --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["TypeofContact"] = st.selectbox(
        "TypeofContact", ["Self Enquiry", "Company Invited"]
    )
with c2:
    user_input["Occupation"] = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
with c3:
    user_input["Gender"] = st.selectbox(
        "Gender", ["Male", "Female"]
    )

# -------- Row 6 (Categorical) --------
c1, c2, c3 = st.columns(3)
with c1:
    user_input["ProductPitched"] = st.selectbox(
        "ProductPitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
with c2:
    user_input["MaritalStatus"] = st.selectbox(
        "MaritalStatus",
        ["Single", "Married", "Divorced"]
    )
with c3:
    user_input["Designation"] = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

st.write("---")

# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------
input_df = pd.DataFrame([user_input])

st.subheader("📊 Customer Input Data")
st.dataframe(input_df, use_container_width=True)

st.divider()

# --------------------------------------------------
# CENTERED PREDICT BUTTON
# --------------------------------------------------
l, c, r = st.columns([3, 2, 3])
with c:
    predict_btn = st.button("🚀 Predict", use_container_width=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_btn:
    input_df = pd.DataFrame([user_input])
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(processed_input)[0][1]
    else:
        prob = 0.0

    st.write("---")

    if prediction == 1:
        st.success("✅ Customer WILL Purchase the Package")
        st.balloons()
    else:
        st.error("❌ Customer will NOT Purchase the Package")

    st.write(f"**Prediction Probability:** {int(prob * 100)}%")
    st.progress(int(prob * 100))
