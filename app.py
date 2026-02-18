import streamlit as st
import joblib

#  Load model and vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('vectorizer.pkl')

st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's Spam or Ham.")

#  Input text from user
user_input = st.text_area("Your Message:")

if st.button("Predict"):
    if user_input.strip() != "":
        #  Transform input using TF-IDF
        X_input = tfidf.transform([user_input])
        
        #  Predict
        prediction = model.predict(X_input)[0]
        # Probability of being Spam (index 1)
        spam_prob = model.predict_proba(X_input)[0][1]
        
        
        
        #  Display result
        if prediction == 1:
            st.error(f"⚠️ This is SPAM!")
        else:
            st.success(f"✅ This is HAM (Not Spam)!")

        #  Display Confidence Bar
        st.write(f"**Spam Confidence:** {spam_prob:.2%}")
        st.progress(spam_prob)  

    else:
        st.warning("Please enter a message to predict.")