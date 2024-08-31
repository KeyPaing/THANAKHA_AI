import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Simbolo-Servicio/Myanmarsar-GPT")
model = AutoModelForCausalLM.from_pretrained("Simbolo-Servicio/Myanmarsar-GPT")

# Load the dataset
df = pd.read_csv('/Users/pyaephyopaing/Desktop/THANAKA AI/Dataset for test chatbot - Sheet1.csv')

# Streamlit UI
st.title("THANAKHA AI")

# Initialize or retrieve conversation history and input state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Display conversation history
for question, answer in st.session_state.history:
    st.write(f"🤔: {question}")
    st.write(f"🤖: {answer}")

# User input
user_input = st.text_input("You:", placeholder="အရေပြားနဲ့ပတ်သက်ပြီးမေးလိုရာမေးနိုင်ပါသည် …", value=st.session_state.user_input, key="input_box")

# Handle submission
if st.button("မေးမည်"):
    if user_input:
        # Find the closest question in the dataset
        matched_row = df[df['Input (Questions)'].str.contains(user_input, na=False)]

        # If a match is found, return the corresponding answer
        if not matched_row.empty:
            answer = matched_row['Result (Answers)'].values[0]
        else:
            # If no match is found, use the model to generate an answer
            input_ids = tokenizer.encode(user_input, return_tensors='pt').to('cpu')
            output = model.generate(input_ids, max_length=50)
            answer = tokenizer.decode(output[0], skip_special_tokens=True)

        # Update conversation history
        st.session_state.history.append((user_input, answer))
        st.session_state.user_input = ""  # Clear the input box by updating session state

        # Display updated conversation history
        for question, answer in st.session_state.history:
            st.write(f"🤔: {question}")
            st.write(f"🤖: {answer}")

# Reset the input box (if needed)
if user_input == "":
    st.session_state.user_input = ""
