import streamlit as st
import joblib
import numpy as np


st.title("Species prediction using iris dataest")
st.write("Choose a model and enter flower measurements to predict the species")

chooseModel = st.selectbox(
    "Select Model",
    (
        "SVM - Binary classfication",
        "SVM - Multiple classification",
        "Logistic Regression - Binary",
        "Logistic Regression - OVR",
        "Logistic Regression - Multinomial"
    )
)

modelMap = {
    "SVM - Binary classfication": "svm_binary_iris.pkl",
    "SVM - Multiple classification": "svm_multi_iris.pkl",
    "Logistic Regression - Binary": "logistic_binary_iris.pkl",
    "Logistic Regression - OVR": "logisticReg_OVR_iris.pkl",
    "Logistic Regression - Multinomial": "logistic_Multinomial_iris.pkl"
}

modelFilename = modelMap[chooseModel]
model = joblib.load(modelFilename)

st.subheader("Enter flower measurements")

sepal_len = st.number_input("Sepal length (cm)", min_value=0.1, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.1, step=0.1)
petal_len = st.number_input("Petal length (cm)", min_value=0.1, step=0.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.1, step=0.1)

if st.button("Predict Species"):
    features = np.array([[sepal_len, sepal_width, petal_len, petal_width]])
    prediction = model.predict(features)[0]

    speciesName = ['setosa', 'versicolor', 'virginica']

    st.success(f"Predicted species: **{speciesName[prediction]}**")



    