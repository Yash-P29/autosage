"""
AutoSage — Export Module
Generate FastAPI inference scripts and Streamlit demo apps for deployment.
"""

def generate_fastapi_script(model_filename: str, features: list) -> str:
    """Generate a FastAPI app script string."""
    fields = "\n".join([f"    {f}: float" for f in features])
    dict_map = ",\n        ".join([f"'{f}': data.{f}" for f in features])
    
    script = f'''from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="AutoSage Model API")

try:
    model = joblib.load("{model_filename}")
except Exception as e:
    print(f"Failed to load model: {{e}}")
    model = None

class InferenceData(BaseModel):
{fields}

@app.post("/predict")
def predict(data: InferenceData):
    if model is None:
        return {{"error": "Model not loaded"}}
    df = pd.DataFrame([{{
        {dict_map}
    }}])
    prediction = model.predict(df)[0]
    return {{"prediction": str(prediction)}}

# Run with: uvicorn app:app --reload
'''
    return script


def generate_streamlit_script(model_filename: str, features: list) -> str:
    """Generate a Streamlit demo app script string."""
    inputs = "\n".join([f"    {f} = st.number_input('{f}', value=0.0)" for f in features])
    dict_map = ",\n        ".join([f"'{f}': {f}" for f in features])
    
    script = f'''import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="AutoSage Model Demo")
st.title("🤖 Model Inference")

@st.cache_resource
def load_model():
    return joblib.load("{model_filename}")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {{e}}")
    st.stop()

with st.form("inference_form"):
{inputs}
    
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([{{
        {dict_map}
    }}])
    prediction = model.predict(df)[0]
    st.success(f"Prediction: {{prediction}}")
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df)[0]
        st.write("Class Probabilities:", dict(enumerate(probas)))

# Run with: streamlit run app.py
'''
    return script
