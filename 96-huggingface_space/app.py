import gradio as gr
import pandas as pd
import joblib
import numpy as np
import datetime as dt

# =========================================================
# Utility Functions
# =========================================================

def replace_unknowns(df):
    """
    Replace 'unknown' string values with NaN so the model's
    preprocessing pipeline can handle them via imputers.
    """
    return df.replace("unknown", np.nan)

# Default value for the "month of contact" input
month_val = float(dt.datetime.now().month)


# =========================================================
# Job Category Mapping (same transformation as training)
# =========================================================

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

# =========================================================
# Model Wrapper (compatible with training pipeline)
# =========================================================

class ModelWrapper:

    """
    Container for:
    - A full sklearn/imb pipeline (preprocessing + model)
    - A decision threshold
    - Optional metadata (evaluation metrics, etc.)

    Used both during training and for deployment in Gradio.
    """

    def __init__(self, pipeline, threshold=0.5, metadata=None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.metadata = metadata if metadata else {}

    def predict_proba(self, X):
        """Return probability of the positive class (y=1)."""
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        """Return binary prediction using the stored threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    @staticmethod
    def load(path):
        """
        Load a saved ModelWrapper (dictionary saved with joblib).
        """
        obj = joblib.load(path)
        return ModelWrapper(
            pipeline=obj["pipeline"],
            threshold=obj["threshold"],
            metadata=obj.get("metadata", {})
        )

# =========================================================
# Load the trained model from HuggingFace directory
# =========================================================


# model_path = "96-huggingface_space/model.pkl"
model_path = "model.pkl" # HF Spaces expects the file at project root
model = ModelWrapper.load(model_path)
threshold = model.threshold # Custom decision threshold

# =========================================================
# Feature Engineering Helpers (must match training pipeline)
# =========================================================

def get_age_bin(age):

    """
    Convert numeric age into the same categorical bins
    used during model training for job x age interactions.
    """

    bins = [18, 30, 45, 60, 100]
    labels = ["18_30", "30_45", "45_60", "60_100"]
    return pd.cut([age], bins=bins, labels=labels)[0]

# =========================================================
# Main Prediction Function (called by Gradio)
# =========================================================

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

    """
    Build a single-row DataFrame with engineered features and run
    the pipeline + threshold to obtain final class prediction.
    """

    # Map raw job value into engineered job category
    job = job_mapping.get(job_raw, "unknown")

    # Interaction feature job × age_bin
    age_bin = get_age_bin(age)
    job_x_age = f"{job}_{age_bin}"

    # Create DataFrame with the same structure as training data
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

    # Predict probability using full pipeline
    proba = model.predict_proba(X)[0]
    # Apply custom threshold from training phase
    pred = 1 if proba >= threshold else 0

    return f"{proba:.4f}", int(pred)

# =========================================================
# Gradio Interface
# =========================================================

contact_options = ["cellular", "telephone", "unknown"]
poutcome_options = ["success", "failure", "nonexistent", "unknown"]
job_options = list(job_mapping.keys())

with gr.Blocks() as demo:
    gr.Markdown("""
    # 📞 Bank Telemarketing Campaign Outcome Predictor:
    This application estimates the **probability** that a customer will subscribe to a term deposit based on campaign and macroeconomic features.
    """)

    # -----------------------------------------------------
    # Macroeconomic Inputs
    # -----------------------------------------------------
    gr.Markdown("## 🏦 Macroeconomic Indicators")
    
    with gr.Row():
        emp_var_rate = gr.Slider(-3.5, 1.5, step=0.1, label="Employment var. rate", value=-0.1, info="Employment variation rate - quarterly indicator")
        cons_price_idx = gr.Slider(92.2, 94.78, step=0.001, label="Consumer Price Index", value=93.798, info="Consumer Price Index - monthly indicator")

    with gr.Row():
        euribor3m = gr.Slider(0.63, 5.05, step=0.001, label="Euribor 3 month rate", value=5.045, info="Euribor 3 month rate - daily indicator")
        nr_employed = gr.Slider(4963, 5229, step=0.1, label="Number of employees", value=5195.8, info="Number of employees - monthly indicator")

    # ---------------------
    # DROPDOWNS
    # ---------------------
    gr.Markdown("## 👤 Customer and Campaign Features")
    with gr.Row():
        contact = gr.Dropdown(contact_options, label="Contact", value="telephone", info="Contact communication type")
        poutcome = gr.Dropdown(poutcome_options, label="Previous outcome", value="nonexistent", info="Outcome of the previous marketing campaign")

    # ---------------------
    # Job, AGE, marital
    # ---------------------
    with gr.Row():
        job = gr.Dropdown(job_options, label="Job", value="technician", info="Customer's job type")
        age = gr.Number(label="Age", value=43, info="Customer's age in years")
    with gr.Row():
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