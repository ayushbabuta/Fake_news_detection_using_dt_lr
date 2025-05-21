import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Title
st.title("Fake News Detection App")

st.write("Enter a news article below, select a model, and click Predict to find out if it's Real or Fake.")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a Machine Learning Model:",
    ("Logistic Regression", "Decision Tree")
)

# Load the selected model
if model_choice == "Logistic Regression":
    with open("model_logistic.pkl", "rb") as file:
        vectorizer, model = pickle.load(file)
else:
    with open("model.pkl", "rb") as file:
        vectorizer, model = pickle.load(file)

# Text input
user_input = st.text_area("News Text:", height=250)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        # Preprocess text
        sentence = re.sub(r'[^\w\s]', '', user_input)
        cleaned_text = ' '.join(
            word.lower() for word in sentence.split()
            if word.lower() not in stopwords.words('english')
        )

        # Transform and predict
        input_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.success("This news seems Real.")
        else:
            st.error("This news seems Fake.")

# Footer
st.markdown("---")
st.caption("Built with love using Streamlit")
