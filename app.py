  
import streamlit as st
import pickle
 


st.title("Predicting Diabetes")
st.header("Input")

def predict():
    try:
        P=st.number_input("Pregnancies")
        G=st.number_input("Glucose")
        B=st.number_input("BloodPressure")
        S=st.number_input("SkinThickness")
        I=st.number_input("Insulin")
        BM=st.number_input("BMI")
        D=st.number_input("DiabetesPedigreeFunction(Based on Family history)")
        A=st.number_input("Age")
        value=[P,G,B,S,I,BM,D,A]
        f='regressor.pickle'
        model=pickle.load(open(f,'rb'))
        if st.button('Predict'):
            p=model.predict([value])
             
            if p[0]==0:
                st.success('You have no signs of Diabetes')
                st.balloons()
            else:
                st.success('You must consulta doctor')
    except:
        st.exception('There is some exception')

    

predict()
