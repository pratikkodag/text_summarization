import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

# Streamlit Page Configuration
st.set_page_config(page_title="Text Summarization", page_icon="📝")

# Title of the Webpage
st.title("Text Summarization Application")

# Load model and tokenizer (make sure the folder path is correct)

    # Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Text input for summarization
input_text = st.text_area("Enter text to summarize", height=300)
preprocess_text = input_text.strip().replace('\n','')
input_text= "summarize:"+ preprocess_text
    # Button to generate summary
if st.button("Generate Summary"):
    if input_text:
        # Encode the input text and prepare the model for generation
        inputs = tokenizer.encode(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            
            # Generate summary
        summary_ids = model.generate(inputs, max_length=100,min_length=20,early_stopping=True)
            
            # Decode the summary and display
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        st.subheader("Generated Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")