import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, 'rb') as f:
        image_data = f.read()
    
    b64_encoded = base64.b64encode(image_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)