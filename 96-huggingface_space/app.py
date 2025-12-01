import gradio as gr
import joblib
import pandas as pd

model = joblib.load("model.pkl")  # use your filename here

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    proba = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    return {
        "Prediction": int(pred),
        "Probability of YES": float(proba)
    }

inputs = [
    gr.Number(label="age"),
    gr.Number(label="campaign"),
    gr.Number(label="pdays"),
    gr.Number(label="previous"),
    # etc...
]

gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="json",
    title="Telemarketing Classifier"
).launch()
