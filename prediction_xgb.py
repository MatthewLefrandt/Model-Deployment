#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encoders
model = joblib.load('C:/Users/Matthew/Downloads/BestModel_XGB.pkl')
gender_encode = joblib.load('C:/Users/Matthew/OneDrive/Documents/Jawaban UTS Model Deployment Matthew Lefrandt 2602107934/gender_encode.pkl')
oneHot_encode_geo = joblib.load('C:/Users/Matthew/OneDrive/Documents/Jawaban UTS Model Deployment Matthew Lefrandt 2602107934/oneHot_encode_geo.pkl')

def main():
    st.title('Churn Model Deployment')

    # Add user input components for 11 features
    credit_score = st.number_input("Credit Score", 0, 10000)
    geography = st.radio("Geography", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 100)
    tenure = st.number_input("Tenure", 0, 100)
    balance = st.number_input("Balance", 0, 100)
    num_of_products = st.number_input("Number of Products", 0, 10)
    has_cr_card = st.radio("Has Credit Card", ["Yes", "No"])
    is_active_member = st.radio("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", 0, 1000000)


    data = {'CreditScore': credit_score, 'Geography': geography, 'Gender': gender,
            'Age': age, 'Tenure': tenure, 'Balance': balance,
            'NumOfProducts': num_of_products, 'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary}

    df = pd.DataFrame([list(data.values())], columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                                'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

    df = df.replace({'Gender': gender_encode})
    cat_geo = df[['Geography']]
    cat_enc_geo = pd.DataFrame(oneHot_encode_geo.transform(cat_geo).toarray(),columns=oneHot_encode_geo.get_feature_names_out())
    df=pd.concat([df,cat_enc_geo], axis=1)
    df=df.drop(['Geography'],axis=1)
    
    if st.button('Make Prediction'):
        result = make_prediction(df_encoded)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    input_array = np.array(features)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

