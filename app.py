import streamlit as st
from transformers import pipeline, set_seed

# Load dialogue generation model
@st.cache_resource
def load_generator():
    generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    return generator

generator = load_generator()
set_seed(42)

st.set_page_config(page_title="🎭 Dialogue Generator", layout="centered")
st.title("🎭 Dialogue Generator")

prompt = st.text_input("💬 Enter a prompt or start of conversation:", value="Hi, how are you?")

if st.button("Generate Dialogue"):
    with st.spinner("Generating response..."):
        output = generator(prompt, max_length=100, num_return_sequences=1)
        response = output[0]['generated_text']
    st.subheader("🗣️ Generated Dialogue")
    st.write(response)
