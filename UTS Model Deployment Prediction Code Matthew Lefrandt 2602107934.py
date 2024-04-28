import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load('C:/Users/Matthew/Downloads/BestModel_XGB.pkl')

def main():
    st.title('Churn Prediction Model Deployment With XGBoost')

    # Add user input components for features
    credit_score = st.slider('Credit Score', min_value=0, max_value=10000, value=5000)
    geography = st.selectbox('Geography', ['Spain', 'France', 'Germany'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', min_value=0, max_value=100, value=30)
    tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
    balance = st.slider('Balance', min_value=0, max_value=100, value=50)
    num_of_products = st.slider('Number of Products', min_value=0, max_value=10, value=1)
    has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
    estimated_salary = st.slider('Estimated Salary', min_value=0, max_value=1000000, value=500000)

    # Convert categorical features to numerical
    geography_mapping = {'Spain': 0, 'France': 1, 'Germany': 2}
    gender_mapping = {'Male': 1, 'Female': 0}
    has_cr_card_mapping = {'Yes': 1, 'No': 0}
    is_active_member_mapping = {'Yes': 1, 'No': 0}

    geography_encoded = geography_mapping[geography]
    gender_encoded = gender_mapping[gender]
    has_cr_card_encoded = has_cr_card_mapping[has_cr_card]
    is_active_member_encoded = is_active_member_mapping[is_active_member]

    if st.button('Make Prediction'):
        features = [credit_score, geography_encoded, gender_encoded, age, tenure, balance, num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

