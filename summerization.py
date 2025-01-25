import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os


st.set_page_config(page_title="Text Summarization", page_icon="📝")


st.title("Text Summarizer")


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

   
input_text = st.text_area("Enter text to summarize", height=300)
preprocess_text = input_text.strip().replace('\n','')
input_text= "summarize:"+ preprocess_text
 
if st.button("Generate Summary"):
    if input_text:
       
        inputs = tokenizer.encode(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
            
           
        summary_ids = model.generate(inputs, max_length=100,min_length=20,early_stopping=True)
            
            
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        st.subheader("Generated Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")