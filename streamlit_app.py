import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.title('Machine Learning single prediction')

st.divider()  

#import data to predict
file_path = './domain_encoded.xlsx'  
domain_encoded = pd.read_excel(file_path)


firstName = st.text_input('Input your FirstName: ')
st.write('Your FirstName is', firstName)

lastName = st.text_input('Input your LastName: ')
st.write('Your LastName is', lastName)

genderradio=st.radio(
    "What's your gender",
    [ "Male:man:","Female:woman:"])

if genderradio == 'Male:man:':
    gender='M'
    st.write('You are a man.')
else:
    gender='F'
    st.write('You are a woman.')

options = domain_encoded['Domain']
domaine = st.selectbox("Select a value", options)
st.write('You selected:', domaine)

colonneNiveau = st.number_input('What is your level of education ?',step=1)
colonneExperience = st.number_input('How many years of experience you have ?',step=1)

training_columns = ['Gender', 'Domain', 'Experience', 'Niveau']

data=[{'firstName':firstName,
      'lastName':lastName,
      'Gender':gender,
      'Domaine':domaine,
      'ColonneNiveau':colonneNiveau,
      'ColonneExperience':colonneExperience}]

df = pd.DataFrame(data)
def min_max_scaling(column):
    try:
        max_values = {'ColonneExperience': 29.0, 'ColonneNiveau': 12}
        max_value = max_values.get(column.name)
        if max_value is None:
            raise ValueError("Maximum value not provided for column: {column.name}")
        
        return (column - 0) / (max_value - 0)
    except TypeError:
        print(f"Skipping normalization for non-numeric column: {column.name}")
        return column
    
columns_to_normalize = ['ColonneExperience','ColonneNiveau']


# Apply Min-Max Scaling to selected columns
for column in columns_to_normalize:
    df[f'Normalized_{column}'] = min_max_scaling(df[column])

#Label the data
weights = {'ColonneExperience': 0.6, 'ColonneNiveau': 0.4}
df['Weighted_Score'] = sum(df[f'Normalized_{col}'] * weights[col] for col in weights)
df['Weighted_Score'] = df['Weighted_Score'].astype(float)
threshold = 0.36
for i, row in df.iterrows():
    if row['Weighted_Score']>=(threshold):
        df.at[i, 'Output'] = 1
    else:
        df.at[i, 'Output'] = 0


domain_mapping = dict(zip(domain_encoded['Domain'], domain_encoded['Domain_encoded']))
df['Domain_encoded'] = df['Domaine'].map(domain_mapping)

def encode_domaine(gender):
    if gender == 'M':
        return 1
    elif gender == 'F':
        return 1
df['Gender'] = df['Gender'].apply(encode_domaine)
df.rename(columns = {'Domain_encoded':'Domain'}, inplace = True) 
df.rename(columns = {'Weighted_Score':'Score'}, inplace = True) 
df.rename(columns = {'Normalized_ColonneExperience':'Experience'}, inplace = True) 
df.drop(['firstName', 'lastName', 'ColonneNiveau', 'ColonneExperience'], axis=1, inplace=True)
df.rename(columns = {'Normalized_ColonneNiveau':'Niveau'}, inplace = True) 

X_test = df.drop(['Output','Score','Domaine'], axis=1)
X_test = X_test[training_columns]
y_test = df['Output']
# Load the trained model
model = joblib.load('model.pkl')


# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_test_imputed = imputer.fit_transform(X_test)
# Use the model to make predictions
predictions = model.predict(X_test)
st.write(f"predictions: {predictions}")

df['Predictions'] = predictions
# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy: {accuracy:.4f}")

df=df.round(2)
st.subheader('Data Table')
st.dataframe(df)
