import streamlit as st
import joblib
import pandas as pd



@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('models/final_model.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'final_model_pipeline.pkl' is in the 'models/' directory.")
        return None


pipeline = load_model()


st.title("Heart Disease Prediction App")
st.markdown("Enter patient details into the sidebar to get a prediction on the likelihood of heart disease.")

st.sidebar.header("Patient Data Input")


def get_user_input():

    # Numerical Inputs
    age = st.sidebar.slider('Age', 20, 80, 55)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (Thalach)', 60, 220, 150)
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0)

    # Categorical Inputs with numerical mapping
    cp_options = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    cp_selection = st.sidebar.selectbox('Chest Pain Type (CP)', options=list(cp_options.keys()))
    cp = cp_options[cp_selection]

    exang_options = {'No': 0, 'Yes': 1}
    exang_selection = st.sidebar.selectbox('Exercise Induced Angina (Exang)', options=list(exang_options.keys()))
    exang = exang_options[exang_selection]

    slope_options = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope_selection = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', options=list(slope_options.keys()))
    slope = slope_options[slope_selection]

    ca = st.sidebar.selectbox('Number of Major Vessels (Ca)', options=[0, 1, 2, 3, 4])

    thal_options = {'Normal': 3, 'Fixed Defect': 6, 'Reversible Defect': 7}
    thal_selection = st.sidebar.selectbox('Thalassemia (Thal)', options=list(thal_options.keys()))
    thal = thal_options[thal_selection]


    user_data = {
        'ca': [ca],
        'thal': [thal],
        'age': [age],
        'cp': [cp],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope]
    }


    features_df = pd.DataFrame(user_data)
    return features_df


user_input_df = get_user_input()

st.subheader("Patient's Input Data")
st.write(user_input_df)


if pipeline:
    if st.sidebar.button("Predict"):
        try:
            prediction = pipeline.predict(user_input_df)
            prediction_proba = pipeline.predict_proba(user_input_df)

            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("High Risk: The model predicts the presence of heart disease.")
            else:
                st.success("Low Risk: The model predicts no evidence of heart disease.")

            st.subheader("Prediction Probability")
            st.write(f"Probability of **Heart Disease**: {prediction_proba[0][1]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Model could not be loaded. Prediction is unavailable.")