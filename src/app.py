import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/llama-7b-base"  # Adjust for different Llama models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_text(prompt):
    encoded_input = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_input)
    generated_ids = outputs.last_hidden_state.argmax(dim=-1)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

st.title("Llama Text Generation App")
user_prompt = st.text_input("Enter your prompt:")

if st.button("Generate Text"):
    if user_prompt:
        generated_text = generate_text(user_prompt)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt.")
