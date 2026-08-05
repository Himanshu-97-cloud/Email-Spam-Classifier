import streamlit as st
import main

st.title("📧 Spam Email Classifier")

st.write(
    "Enter an email message below and the model will predict whether it is Spam or Ham."
)

email = st.text_area("Enter Email Text")

if st.button("Predict"):

    if email.strip() == "":

        st.warning("Please enter an email.")

    else:

        prediction = main.predict(email)

        if prediction == 1:

            st.error("🚨 Spam Email")

        else:

            st.success("✅ Ham (Not Spam)")