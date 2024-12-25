import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model with error handling
try:
    model = joblib.load('random_forest_model_1.pkl')
except FileNotFoundError:
    st.error('Model file not found. Please ensure the model is in the correct directory.')
except Exception as e:
    st.error(f"An error occurred: {e}")

# User input function
def user_input_features():
    st.sidebar.subheader('Please provide your details:')
    # Input fields (as per your code)
    # ...
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
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display input data
st.subheader('User Input parameters')
st.write(input_df)

# Predict and display result
try:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction')
    heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
    if prediction == 1:
        st.markdown(f"<span style='color:red;font-weight:bold;'>{heart_disease[prediction]}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green;font-weight:bold;'>{heart_disease[prediction]}</span>", unsafe_allow_html=True)
    
    # Display probabilities
    st.subheader('Prediction Probability')
    st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")

    # Provide recommendations
    if prediction == 0:  # No Heart Disease
        st.write("You have a low risk of heart disease. Continue maintaining a healthy lifestyle.")
    else:
        st.subheader('Additional Recommendations')
        st.write("Further evaluation with your doctor is recommended. Suggested tests:")
        st.write("- Electrocardiogram (ECG)\n- Echocardiogram\n- Stress Test\n- Coronary Angiography\n- Cardiac MRI")
    
        # Custom recommendations based on input
        recommendations = get_recommendations(input_df)
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
