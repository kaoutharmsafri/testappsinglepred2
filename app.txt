import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Define Streamlit UI
st.title('Machine Learning multiple prediction')

#import data to predict
data = {
    'ID': [1, 2,3],
    'Nom': ['SADAK', 'ZAKARIA','ASLI'],
    'Prénom': ['Marouane', 'Mohammed', 'Othmane'],
    'Fonction': [" ingénieur en mécanique : spécialité aéronautique automobile", " ingénieur en mécanique", 'Ingénieur Industriel'],
    'Gender': ['M', 'M', 'M'],
    'Domaine': ['Ingénieur Mécanique', 'Ingénieur Mécanique', 'Ingénieur Industriel'],
    'ColonneNiveau': [6, 6,5],
    'ColonneExperience': [5,5,2]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Display the DataFrame in Streamlit
st.write("Sample DataFrame:")
st.dataframe(df)

training_columns = ['Gender', 'Domain', 'Experience', 'Niveau']

# Min-Max Scaling function
def min_max_scaling(column):
    try:
        return (column - column.min()) / (column.max() - column.min())
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

df=df.round(2)

df['Domaine'] = df['Domaine'].str.lower()

file_path = './domain_encoded.xlsx'  
domain_encoded = pd.read_excel(file_path)
domain_mapping = dict(zip(domain_encoded['Domain'], domain_encoded['Domain_encoded']))
df['Domain_encoded'] = df['Domaine'].map(domain_mapping)

def encode_domaine(gender):
    if gender == 'M':
        return 1
    elif gender == 'F':
        return 1
df['Gender'] = df['Gender'].apply(encode_domaine)

# rename and encode the data
df.rename(columns = {'Domain_encoded':'Domain'}, inplace = True) 
df.rename(columns = {'Domaine':'DomainTitle'}, inplace = True) 
df.rename(columns = {'Normalized_ColonneExperience':'Experience'}, inplace = True) 
label_encoder = LabelEncoder()
# df['Gender'] = label_encoder.fit_transform(df['Gender'])
df.drop(['ID', 'Nom', 'Prénom', 'Fonction', 'ColonneNiveau', 'ColonneExperience'], axis=1, inplace=True)
df.rename(columns = {'Normalized_ColonneNiveau':'Niveau'}, inplace = True) 

# Split the data into features (X) and target (y)
X_test = df.drop(['Output','Weighted_Score','DomainTitle'], axis=1)
X_test = X_test[training_columns]
y_test = df['Output']

# Load the trained model
model = joblib.load('model.pkl')

# Use the model to make predictions
predictions = model.predict(X_test)
st.write(f"predictions: {predictions}")

df['Predictions'] = predictions
# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
st.write(f"Accuracy: {accuracy:.4f}")

st.dataframe(df.drop(['Weighted_Score'], axis=1))
