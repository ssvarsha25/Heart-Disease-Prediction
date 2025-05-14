import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model_1.pkl')

# Define the input fields for user with descriptions
def user_input_features():
    st.sidebar.subheader('Please provide your details:')
    
    age = st.sidebar.slider(
        'Age',
        1,
        120,
        25,
        help='Select your age. The range is from 1 to 120 years.'
    )
    
    sex = st.sidebar.selectbox(
        'Sex',
        (0, 1),
        format_func=lambda x: 'Male' if x == 1 else 'Female',
        help='Select your gender. 0 indicates Female and 1 indicates Male.'
    )
    
    cp = st.sidebar.selectbox(
        'Chest Pain Type',
        (0, 1, 2, 3),
        format_func=lambda x: {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal Pain', 3: 'Asymptomatic'}[x],
        help='Select the type of chest pain you experienced.'
    )
    
    trestbps = st.sidebar.slider(
        'Resting Blood Pressure',
        50,
        200,
        120,
        help='Select your resting blood pressure in mm Hg. The range is from 50 to 200 mm Hg.'
    )
    
    chol = st.sidebar.slider(
        'Serum Cholesterol in mg/dl',
        100,
        400,
        200,
        help='Select your serum cholesterol level in mg/dl. The range is from 100 to 400 mg/dl.'
    )
    
    fbs = st.sidebar.selectbox(
        'Fasting Blood Sugar > 120 mg/dl',
        (0, 1),
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help='Select whether your fasting blood sugar is greater than 120 mg/dl.'
    )
    
    restecg = st.sidebar.selectbox(
        'Resting Electrocardiographic Results',
        (0, 1, 2),
        format_func=lambda x: {0: 'Normal', 1: 'Having ST-T Wave Abnormality', 2: 'Showing Probable or Definite Left Ventricular Hypertrophy'}[x],
        help='Select the result of your resting electrocardiogram.'
    )
    
    thalach = st.sidebar.slider(
        'Maximum Heart Rate Achieved',
        50,
        220,
        150,
        help='Select your maximum heart rate achieved during exercise.'
    )
    
    exang = st.sidebar.selectbox(
        'Exercise Induced Angina',
        (0, 1),
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help='Select whether you experience angina induced by exercise.'
    )
    
    oldpeak = st.sidebar.slider(
        'ST Depression Induced by Exercise',
        0.0,
        10.0,
        1.0,
        step=0.1,
        help='Select the amount of ST depression induced by exercise relative to rest.'
    )
    
    slope = st.sidebar.selectbox(
        'Slope of the Peak Exercise ST Segment',
        (0, 1, 2),
        format_func=lambda x: {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}[x],
        help='Select the slope of the peak exercise ST segment.'
    )
    
    ca = st.sidebar.slider(
        'Number of Major Vessels Colored by Fluoroscopy',
        0,
        3,
        0,
        help='Select the number of major vessels (0-3) colored by fluoroscopy.'
    )
    
    thal = st.sidebar.selectbox(
        'Thalassemia',
        (1, 2, 3),
        format_func=lambda x: {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}[x],
        help='Select the type of thalassemia.'
    )

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(input_df)

# Predict and display the result
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction])

# Provide recommendations and future probability for 'No Heart Disease' prediction
if prediction == 0:  # No Heart Disease
    st.subheader('Prediction Probability')
    st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
    
    # Risk evaluation and recommendations for future
    st.subheader('Future Risk Evaluation')
    st.write("Based on your inputs, you currently have a low risk of heart disease.")
    st.write("Maintaining a healthy lifestyle and regular checkups are recommended.")

else:  # Heart Disease
    # Risk detected, recommend further tests
    st.subheader('Recommended Tests')
    st.write("Since there are indications of heart disease, it is advised to undergo the following tests for further evaluation:")
    st.write("""
    - Electrocardiogram (ECG or EKG)
    - Echocardiogram
    - Stress Test
    - Coronary Angiography
    - Cardiac MRI
    """)
    
    # Provide lifestyle recommendations based on user input
    def get_recommendations(df):
        recommendations = []
        
        if df['trestbps'][0] > 120:
            recommendations.append("Consider monitoring and managing your blood pressure.")
        
        if df['chol'][0] > 200:
            recommendations.append("High cholesterol detected. Consider a diet low in saturated fats and cholesterol.")
        
        if df['fbs'][0] == 1:
            recommendations.append("High fasting blood sugar detected. Monitor your blood sugar levels and consider consulting a doctor.")
        
        if df['exang'][0] == 1:
            recommendations.append("Exercise induced angina detected. Consider consulting a doctor for a detailed examination.")
        
        return recommendations
    
    recommendations = get_recommendations(input_df)
    
    if recommendations:
        st.subheader('Additional Recommendations')
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.write("No additional recommendations based on the inputs.")
