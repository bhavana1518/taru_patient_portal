import streamlit as st
from transformers import pipeline
import torch

st.title("🧬 Lab Result Summarizer Demo")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

def summarize_lab(test, value, unit):
    model = load_model()
    prompt = f"Explain this lab test in simple terms: {test} result is {value}{unit}."
    result = model(prompt, max_length=60, num_return_sequences=1)
    return result[0]["generated_text"]

test = st.text_input("Lab Test", "Hemoglobin")
value = st.text_input("Value", "13.5")
unit = st.text_input("Unit", "g/dL")

if st.button("Summarize"):
    summary = summarize_lab(test, value, unit)
    st.write("### AI Summary")
    st.write(summary)
