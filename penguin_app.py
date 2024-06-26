import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
  penguins=model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
  penguins=penguins[0]
  if penguins == 1:
    return 'Adelie'
  elif penguins == 2:
    return 'Chinstrap'
  else:
    return 'Gentoo'

st.sidebar.title('Penguins Species Prediction App')
blm=st.sidebar.slider('Bill length in mm',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bdm=st.sidebar.slider('Bill depth in mm',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
flm=st.sidebar.slider('Flipper length in mm',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
bmg=st.sidebar.slider('Body mass in g',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))
sex1=st.sidebar.selectbox('Sex',('Male','Female'))
if sex1 == 'Male':
  sex=0
else:
  sex=0
island1=st.sidebar.selectbox('Island',('Biscoe','Dream','Torgerson'))
if island1 == 'Biscoe':
  island=0
elif island1 == 'Dream':
  island=1
else:
  island=2
classifier=st.sidebar.selectbox('Classifier',('Logistic Regression','Support Vector Machine','Random Forest Classifier'))
if st.sidebar.button('Predict'):
  if classifier == 'Logistic Regression':
    penguin_type=predicton(log_reg,island,blm,bdm,flm,bmg,sex)
    score=log_reg.score(X_train,y_train)
  elif classifier == 'Support Vector Machine':
    penguin_type=prediction(svc_model,island,blm,bdm,flm,bmg,sex)
    score=svc_model.score(X_train,y_train)
  elif classifier == 'Random Forest Classifier':
    penguin_type=prediction(rf_clf,island,blm,bdm,flm,bmg,sex)
    score=rf_clf.score(X_train,y_train)
  st.write('Species Predicted --> ',penguin_type)
  st.write('Model Accuracy --> ',score)
