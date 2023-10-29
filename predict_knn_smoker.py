
import streamlit as st
# import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

#Load model
with open('knn_isSmoker.pkl', 'rb') as file:
    # Load the data from the file
    model, smoker_encoder, region_encoder ,sex_encoder = pickle.load(file)

st.title("Smoker Prediction")

# ['female' 'male']
# ['southwest' 'southeast' 'northwest' 'northeast']
# ['yes' 'no']

# Get user input for each variable
sex_input = st.selectbox('Sex:', ['female', 'male'])
region_input = st.selectbox('Region:', ['southwest', 'southeast','northwest','northeast'])
age_input = st.number_input('Age (18 to 64):', min_value=18, max_value=64)
bmi_input = st.number_input('BMI (15 to 54):', min_value=15, max_value=54)
children_input = st.number_input('Children (0 to 6):', min_value=0, max_value=6)
charges_input = st.number_input('Charges (1000 to 70,000):', min_value=1000, max_value=70000)

# Index(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'], dtype='object')

# Create a DataFrame with user input
x_new = pd.DataFrame({
    'age': [age_input],
    'sex': [sex_input],
    'bmi': [bmi_input],
    'children': [children_input],
    'region': [region_input],
    'charges': [charges_input]
})

# Encoding
x_new['sex'] = sex_encoder.transform(x_new['sex'])
x_new['region'] = region_encoder.transform(x_new['region'])

# Prediction
y_pred_new = model.predict(x_new)
result = smoker_encoder.inverse_transform(y_pred_new)

# Display result
st.subheader('Prediction Result:')
st.write(f'Predicted Smoker: {result[0]}')
