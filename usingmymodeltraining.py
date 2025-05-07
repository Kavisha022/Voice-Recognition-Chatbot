import streamlit as st
st.set_page_config(page_title="Trained Voice QA Bot", layout="centered")
import speech_recognition as sr
import pyttsx3
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from langdetect import detect
import os
os.environ["STREAMLIT_DISABLE_FILE_WATCHING"] = "1"


# Load trained model
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("./t5_qa_model")
    model = T5ForConditionalGeneration.from_pretrained("./t5_qa_model")
    return tokenizer, model

tokenizer, model = load_model()

# Text_to_speech
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

# Voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening (max 5 sec)...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Sorry, I couldn't understand that."
    
# Answer generation
def generate_answer(question):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(input_ids, max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Streamlit UI
# st.set_page_config("Trained Voice QA Bot", layout="centered")
st.title("Ask Your Trained Data Science Bot (Voice)")

if st.button("Ask Now"):
    question = get_voice_input()
    st.subheader("You asked: ")
    st.write(question)

    try:
        language = detect(question)
        if language != "en":
            st.warning("Please ask in English.")
            speak("Please ask in English.")
        elif "Sorry" in question:
            st.warning("Could not inderstand input.")
            speak("Could not understand.")
        else:
            answer = generate_answer(question)
            st.subheader("Answer: ")
            st.write(answer)
            speak(answer)
    except:
        st.error("Something went wrong.")
        speak("Something went wrong.")
