from google import genai
import streamlit as st
import pandas as pd
import json
from google import genai


# Paste your actual key here just for this test
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

print("Models available to this API key:")
for model in client.models.list():
    # Only print models that can generate text
    if "generateContent" in model.supported_actions:
        print(model.name)