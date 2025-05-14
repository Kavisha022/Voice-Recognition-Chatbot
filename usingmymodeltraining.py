import streamlit as st
st.set_page_config(page_title="Trained Voice QA Bot", layout="centered")

import speech_recognition as sr
import pyttsx3
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langdetect import detect
import torch
import os

# Optional: Improves performance on file changes
os.environ["STREAMLIT_DISABLE_FILE_WATCHING"] = "1"

# Load model
@st.cache_resource
def load_model():
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            "./t5_base_qa_model"  ##############################################
        )
        model = T5ForConditionalGeneration.from_pretrained(
            "./t5_base_qa_model"  ###############################################
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()

# Text-to-speech
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
        try:
            audio = recognizer.listen(source, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.RequestError as e:
            return f"API error: {e}"
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except Exception as e:
            return f"Unexpected error: {e}"

# Answer generation
def generate_answer(question):
    try:
        input_text = f"question: {question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        output_ids = model.generate(input_ids, max_length=128)  ############################################### 64 0r 128
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Streamlit UI
st.title("Ask Your Trained Data Science Bot (Voice)")

if st.button("Ask Now"):
    question = get_voice_input()
    st.subheader("You asked:")
    st.write(question)

    try:
        language = detect(question)
        if language != "en":
            st.warning("Please ask in English.")
            speak("Please ask in English.")
        elif "Sorry" in question or "error" in question.lower():
            st.warning("Could not understand input.")
            speak("Could not understand.")
        else:
            answer = generate_answer(question)
            st.subheader("Answer:")
            st.write(answer)
            speak(answer)
    except Exception as e:
        st.error("Something went wrong.")
        speak("Something went wrong.")
