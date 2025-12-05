import gradio as gr
import pandas as pd
import joblib
import numpy as np
import datetime as dt

def replace_unknowns(df):
    return df.replace("unknown", np.nan)

month_val = float(dt.datetime.now().month)

# -----------------------------
# Job mapping
# -----------------------------
job_mapping = {
    "admin.": "admin",
    "blue-collar": "blue_collar",
    "technician": "technician",
    "services": "services_group",
    "housemaid": "services_group",
    "management": "management",
    "retired": "no_labor_force",
    "student": "no_labor_force",
    "unemployed": "no_labor_force",
    "entrepreneur": "self_employed_group",
    "self-employed": "self_employed_group",
    "unknown": "unknown"
}

# -----------------------------
# Model Wrapper
# -----------------------------
class ModelWrapper:
    def __init__(self, pipeline, threshold=0.5, metadata=None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.metadata = metadata if metadata else {}

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    @staticmethod
    def load(path):
        obj = joblib.load(path)
        return ModelWrapper(
            pipeline=obj["pipeline"],
            threshold=obj["threshold"],
            metadata=obj.get("metadata", {})
        )

# -----------------------------
# Load model
# -----------------------------
# model_path = "96-huggingface_space/model.pkl"
model_path = "model.pkl"
model = ModelWrapper.load(model_path)
threshold = model.threshold

def get_age_bin(age):
    bins = [18, 30, 45, 60, 100]
    labels = ["18_30", "30_45", "45_60", "60_100"]
    return pd.cut([age], bins=bins, labels=labels)[0]

# -----------------------------
# Prediction function
# -----------------------------
def predict(
    emp_var_rate,
    cons_price_idx,
    euribor3m,
    nr_employed,
    contact_val,
    poutcome_val,
    job_raw,
    age,
    pdays,
    previous,
    month):

    job = job_mapping.get(job_raw, "unknown")

    # engineered features
    age_bin = get_age_bin(age)
    job_x_age = f"{job}_{age_bin}"

    X = pd.DataFrame([{
        "pdays": pdays,
        "previous": previous,
        "month_cos": np.cos(2 * np.pi * float(month) / 12),
        "age_emp_rate": float(age) * float(emp_var_rate),
        "euribor_nrm": float(euribor3m) / (float(nr_employed) + 1),
        "job": job,
        "contact": contact_val,
        "poutcome": poutcome_val,
        "job_x_age": job_x_age,
        "emp_var_rate": emp_var_rate,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed,
        "cons_price_idx": cons_price_idx
    }])

    proba = model.predict_proba(X)[0]
    # use the model's threshold
    pred = 1 if proba >= threshold else 0

    return f"{proba:.4f}", int(pred)

# -----------------------------
# Gradio Interface
# -----------------------------
contact_options = ["cellular", "telephone", "unknown"]
poutcome_options = ["success", "failure", "nonexistent", "unknown"]
job_options = list(job_mapping.keys())
marital_options = ["single", "married", "divorced", "unknown"]

with gr.Blocks() as demo:
    gr.Markdown("""# 📞 Predicting the Success of Bank Telemarketing Campaigns
                This application predicts the probability of a customer subscribing to a term deposit based on various features using a pre-trained classification model.""")

    # ---------------------
    # 2×2 grid for SLIDERS
    # ---------------------
    gr.Markdown("## Macroeconomic Indicators")
    with gr.Row():
        emp_var_rate = gr.Slider(-3.5, 1.5, step=0.1, label="Employment var. rate", value=-0.1, info="Employment variation rate - quarterly indicator")
        cons_price_idx = gr.Slider(92.2, 94.78, step=0.001, label="Consumer Price Index", value=93.798, info="Consumer Price Index - monthly indicator")

    with gr.Row():
        euribor3m = gr.Slider(0.63, 5.05, step=0.001, label="Euribor 3 month rate", value=5.045, info="Euribor 3 month rate - daily indicator")
        nr_employed = gr.Slider(4963, 5229, step=0.1, label="Number of employees", value=5195.8, info="Number of employees - monthly indicator")

    # ---------------------
    # DROPDOWNS
    # ---------------------
    gr.Markdown("## Customer and Campaign Features")
    with gr.Row():
        contact = gr.Dropdown(contact_options, label="Contact", value="telephone", info="Contact communication type")
        poutcome = gr.Dropdown(poutcome_options, label="Previous outcome", value="nonexistent", info="Outcome of the previous marketing campaign")

    # ---------------------
    # Job, AGE, marital
    # ---------------------
    with gr.Row():
        job = gr.Dropdown(job_options, label="Job", value="technician", info="Customer's job type")
        age = gr.Number(label="Age", value=43, info="Customer's age in years")  
        marital = gr.Dropdown(marital_options, label="Marital status", value="single", info="Customer's marital status")
        pdays = gr.Number(label="Days since last contact", value=0, info="Number of days since the last contact from a previous campaign")
        previous = gr.Number(label="Number of contacts before this campaign", value=0, info="Number of contacts performed before this campaign for this customer")
        month = gr.Number(label="Month of contact (1-12)", value=month_val, info="Month when the contact was made")

    # ---------------------
    # Output
    # ---------------------
    proba_output = gr.Textbox(label="Predicted Probability")
    class_output = gr.Textbox(label="Predicted Class")

    submit = gr.Button("Predict")

    submit.click(
        predict,
        inputs=[
            emp_var_rate,
            cons_price_idx,
            euribor3m,
            nr_employed,
            contact,
            poutcome,
            job,
            age,
            pdays,
            previous,
            month   
        ],
        outputs=[proba_output, class_output]
    )

if __name__ == "__main__":
    demo.launch()
