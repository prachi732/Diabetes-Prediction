import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import streamlit as st

#Create a title and sub-title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python
""")

#Open display image


#Get the data
df = pd.read_csv('diabetes.csv')

#Set a subheader
st.subheader('Data information:')

#Show the data as a table
st.dataframe(df)

#Show statistics on the data
st.write(df.describe())

#Show the data as a chart


#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#Split the data set into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


#Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    #Store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age,
    }
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])

    return features


#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)
st.subheader("Confusion Matrix")
st.text('Model Report:\n ' + classification_report(Y_test,RandomForestClassifier.predict(X_test)))

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(
    str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) +
    '%')

#Store the models predictions in a variable


#Set a subheader and display the classification

submit = st.button('Predict')

if submit:
  prediction = RandomForestClassifier.predict(user_input)
  if prediction == 0:
    st.subheader('Congratulations You are not diabetic')
            
  else:
    st.subheader("It's better to consult a Doctor")      
            
  
        
  

        
            

 
        
