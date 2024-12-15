import streamlit as st
import pickle
import os

# Paths for the model and vectorization object
project_dir=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(project_dir,"artifacts/models/etc_Count_Vector.pkl")
VECTOR_PATH = os.path.join(project_dir,"artifacts/Count_Vector.pickle")

# Load the model and vectorization object
def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer(MODEL_PATH, VECTOR_PATH)

# Streamlit app
st.title("SMS Spam Classifier")
st.write("Enter an SMS message below to check if it's Spam or Ham.")

# Input text from the user
input_text = st.text_area("SMS Message", placeholder="Type your message here...")

# Predict button
if st.button("Classify Message"):
    if input_text.strip():
        # Vectorize the input text
        vectorized_text = vectorizer.transform([input_text])

        # Make prediction
        prediction = model.predict(vectorized_text)[0]

        # Display result
        if prediction == 1:
            st.success("This message is classified as **Spam**.")
        else:
            st.info("This message is classified as **Ham**.")
    else:
        st.warning("Please enter a valid message to classify.")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name](#).")
