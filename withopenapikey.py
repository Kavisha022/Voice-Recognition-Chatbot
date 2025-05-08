import streamlit as st
import speech_recognition as sr
import pyttsx3
import openai
from langdetect import detect

# Set your OpenAI API key
openai.api_key = " "
# Streamlit app setup
st.set_page_config(page_title="Data Science Voice Q&A", layout="centered")
st.title("Ask Data Science Questions by Voice")

# Text to speech function
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Voice to text function
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your question (max 5 seconds)...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, I couldn't understand the audio."

# Send question to GPT model
def ask_openai(question):
    prompt = f"You are a helpful data science tutor. Answer the following question clearly and briefly:\n{question}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Failed to get a response from OpenAI."

# Button to start
if st.button("Ask your Data Science Question"):
    question = get_voice_input()
    st.subheader("You asked:")
    st.write(question)

    try:
        if detect(question) != "en":
            st.warning("Please ask your question in English.")
            speak("Please ask your question in English.")
        elif "Sorry" in question:
            st.warning("Could not understand your voice input.")
            speak("Sorry, I couldn't understand your question.")
        else:
            answer = ask_openai(question)
            st.subheader("Chatbot's Answer:")
            st.write(answer)
            speak(answer)
    except:
        st.error("Something went wrong.")
        speak("Something went wrong.")
