import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from langdetect import detect

# Set up the Streamlit app
st.set_page_config(page_title="Data Science Voice Q&A Bot", layout="centered")
st.title("Ask Data Science Questions by Voice")

# Load the QA pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/tinyroberta-squad2")

qa_pipeline = load_qa_pipeline()

# Example context about data science – expand this as needed
DATA_SCIENCE_CONTEXT = """
Data science involves statistics, programming, and domain knowledge to extract insights from data.
Supervised learning uses labeled datasets. Common algorithms include:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- K-Nearest Neighbors
- Neural Networks

Unsupervised learning includes clustering (e.g., K-Means) and dimensionality reduction (e.g., PCA).
Deep learning is a subset using multi-layer neural networks and tools like TensorFlow and PyTorch.

Gradient Descent is an optimization algorithm to minimize loss by updating weights.
Overfitting occurs when a model learns the training data too well and fails to generalize.
Cross-validation is used to evaluate model performance on unseen data.
"""


# Text-to-speech function
def speak(text):
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.say(text)
    tts_engine.runAndWait()

# Voice-to-text function
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your question (max 5 seconds)...")
        audio = recognizer.listen(source, phrase_time_limit=5)

    try:
        question = recognizer.recognize_google(audio)
        return question
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Generate answer using QA model
def generate_answer(question):
    result = qa_pipeline(question=question, context=DATA_SCIENCE_CONTEXT)
    return result["answer"]

# Button click to start voice capture
if st.button("Ask your Data Science Question"):
    question = get_voice_input()
    st.subheader("You asked:")
    st.write(question)

    try:
        language = detect(question)
        if language != "en":
            st.warning("Please ask your question in English.")
            speak("Please ask your question in English.")
        elif "Sorry" in question:
            st.warning("Could not understand your voice input.")
            speak("Sorry, I couldn't understand your question.")
        else:
            answer = generate_answer(question)
            st.subheader("Chatbot's Answer:")
            st.write(answer)
            speak(answer)
    except Exception as e:
        st.error("Something went wrong. Try again.")
        speak("Sorry, something went wrong.")
