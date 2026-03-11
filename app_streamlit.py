import streamlit as st
from utils.predict import predict_svm

st.title("Vietnamese Hate Speech Detection")

st.write("Demo sentiment classification using TF-IDF + SVM model.")

# Input box
user_input = st.text_area("Enter Vietnamese comment")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prediction = predict_svm(user_input)

        st.subheader("Prediction:")
        st.success(prediction)