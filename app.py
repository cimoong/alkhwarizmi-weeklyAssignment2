import streamlit as st
import pandas as pd
from tensorflow.python.keras.models import load_model

# Load the trained model
model = load_model('deepnwide.h5')

# Define the predict function
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}
    features = pd.DataFrame(data, index=[0])
    prediction = model.predict(features)
    return prediction[0]

# Create a Streamlit app
def app():
    st.title('Diabetes Prediction')

    st.write('Enter the following details to predict whether you have diabetes or not.')

    # Create input fields for the user to enter the details
    pregnancies = st.slider('Pregnancies', 0, 17, 0)
    glucose = st.slider('Glucose', 0, 199, 70)
    blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
    insulin = st.slider('Insulin', 0, 846, 79)
    bmi = st.slider('BMI', 0.0, 67.1, 20.0)
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.slider('Age', 21, 81, 33)

    # Call the predict function and display the result
    if st.button('Predict'):
        result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        if (result == 0).all():
            st.write('You do not have diabetes.')
        else:
            st.write('You have diabetes.')

if __name__ == '__main__':
    app()
