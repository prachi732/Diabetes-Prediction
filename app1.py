import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st




df = pd.read_csv('diabetes.csv')


X = df.iloc[:, 0:8].values  
Y = df.iloc[:, -1].values  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


def get_user_input():
    pregnancies = st.sidebar.slider('Number of times Pregnant', 0, 15, 1)  
    glucose = st.sidebar.slider('Glucose Level', 0, 199, 110) 
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 140, 72) 
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23) 
    insulin = st.sidebar.slider('Insulin', 0, 126, 100) 
    bmi = st.sidebar.slider('BMI', 0.0, 50.0, 21.5) 
    dpf = st.sidebar.slider('DPF', 0.0, 2.49, 0.3725) 
    age = st.sidebar.slider('AGE', 18, 99, 30) 

    # Kullanıcıdan alınan değerlerin bir sözlük (dictionary) yapısında anahtar-değer (key-value) çiftleri şeklinde kayıt altına alınması
    user_data = {'pregnancies': pregnancies, 'glucose': glucose, 'blood_pressure': blood_pressure, 'skin_thickness': skin_thickness, 'insulin': insulin, 'bmi': bmi, 'dpf': dpf, 'age': age}
    # Kullanıcı verisinin dataframe'e dönüştürülmesi
    features = pd.DataFrame(user_data, index=[0])
    return features

# Kullanıcının girdiği değerleri bir değişkende tutmak
user_input = get_user_input()  # user_input değişkeni kullanıcı girdilerini görüntülemek için kullanılacak

# Web uygulamasına alt başlık oluşturma ve kullanıcı girdilerini görüntüleme
st.subheader('INPUT:')
st.write(user_input)

# Yapay Zeka modelini oluşturup eğitmek
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Web uygulamasına alt başlık oluşturma ve model metriklerini (performansını) görüntüleme

# Modeli Y_test veri setine göre test eder ve RandomForestClassifier modeline X_test veri setini vererek, Y_test'deki değerleri tahmin etme doğruluk puanını belirler
st.write('%' + str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100))  # yüzdelik değer elde etmek için 100 ile çarpıldı


prediction = RandomForestClassifier.predict(user_input)

# Web uygulamasına alt başlık oluşturma ve sınıflandırmayı (şeker hastası mı değil mi) gösterme

st.write(prediction)
