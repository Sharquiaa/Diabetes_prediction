import  streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from plotly.tools import FigureFactory as FF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()

print(diabetes_dataset)

df = pd.read_csv('diabetes.csv')
st.title('Diabetes Prediction Model')
st.image("h1.jpg")
st.write('**Dataset of Diabetic patients**')
st.write(df)

#charts
st.write('**Glucose Data of all the Patients**')

st.bar_chart(df['Glucose'])


#comparison with Age and Glucose
st.write('Comparison with Age and Glucose')
chart_data = pd.DataFrame(
    np.random.randn(50,2), 
    columns=["Glucose","Age"])
st.bar_chart(chart_data)
st.image("image1.png")


dd = diabetes_dataset.describe()
ovc = diabetes_dataset['Outcome'].value_counts()
groupoutcomemean = diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

#standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

#train test data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, stratify=Y, random_state=2)

#training the model
classifier = LogisticRegression()
#st.write('Predicting the input data using Support Vector Machine')   
d = '<p style="font-family:Courier; color:Green; font-size: 30px;"><b>Predicting the input data using Logistic Regression :</b></p>'
st.markdown(d, unsafe_allow_html=True)

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train) 

#accuracy score of the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)
st.write('Accuracy score of the training data:', training_data_accuracy)


#accuracy score of the testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the testing data:', testing_data_accuracy)
st.write('Accuracy score of the testing data:', testing_data_accuracy)

#predictive system
st.write('Input Data')
input_data =(8,181,68,36,495,30.1,0.615,60)
print(input_data)
st.write(input_data)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict_proba(std_data)
print(prediction)
st.write('Prediction for the given input data:')
pred = prediction[0][1]
print(pred)
a = pred*100
st.write('Probability of you having Diabetes is:')
st.write(a,'**%**')
st.image("image3.jpg")