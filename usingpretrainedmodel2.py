import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from langdetect import detect

# Set up Streamlit app
st.set_page_config(page_title="Data Science Voice Q&A Bot", layout="centered")
st.title("🎙️ Ask Your Data Science Question by Voice")

# Load FLAN-T5 model for Q&A
@st.cache_resource
def load_chat_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chat_model = load_chat_model()

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

# Generate answer using chat model
def generate_answer(question):
    prompt = f"Answer the following data science question:\n{question}"
    response = chat_model(prompt, max_length=100, do_sample=True)[0]["generated_text"]
    return response

# Button to trigger voice input
if st.button("🎤 Ask Your Question"):
    question = get_voice_input()
    st.subheader("🗣️ You asked:")
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
            st.subheader("🤖 Chatbot's Answer:")
            st.write(answer)
            speak(answer)
    except Exception as e:
        st.error("Something went wrong. Try again.")
        speak("Sorry, something went wrong.")
