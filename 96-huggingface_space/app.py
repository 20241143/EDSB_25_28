import gradio as gr
import pandas as pd

# ----------------------------------------------------
# DUMMY PREDICTOR (replace this later with your model)
# ----------------------------------------------------
def predict_deposit(age, job, marital, education, campaign, previous, contact, month, duration):
    """
    Dummy function — replace with your actual model.predict().
    Right now it returns a random probability so the UI works.
    """

    # Example simple rule for demo only — REMOVE later
    prob_yes = min(0.95, (campaign * 0.02) + (previous * 0.05))
    prob_no = 1 - prob_yes

    prediction = "YES" if prob_yes > 0.5 else "NO"

    return {
        "Prediction": prediction,
        "Probability YES": prob_yes,
        "Probability NO": prob_no
    }


# ----------------------------------------------------
# GRADIO UI
# ----------------------------------------------------

with gr.Blocks(title="Bank Telemarketing Predictor") as demo:

    gr.Markdown("# 📞 Bank Telemarketing Predictor")
    gr.Markdown(
        "Preencha os campos abaixo para prever se um cliente irá subscrever um depósito a prazo."
    )

    with gr.Row():

        with gr.Column():
            age = gr.Slider(18, 95, value=25, label="Age")
            job = gr.Dropdown(
                ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"],
                label="Job"
            )
            marital = gr.Dropdown(["single", "married", "divorced", "unknown"], label="Marital Status")
            education = gr.Dropdown(["primary", "secondary", "tertiary", "unknown"], label="Education")

        with gr.Column():
            campaign = gr.Number(label="Number of Contacts (Campaign)", value=1)
            previous = gr.Number(label="Previous Contacts", value=0)
            contact = gr.Dropdown(["cellular", "telephone"], label="Contact Type")
            month = gr.Dropdown(
                ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
                label="Month"
            )
            duration = gr.Number(label="Call Duration (secs)", value=100)

    predict_btn = gr.Button("Predict")

    output = gr.Json(label="Prediction Output")

    predict_btn.click(
        fn=predict_deposit,
        inputs=[age, job, marital, education, campaign, previous, contact, month, duration],
        outputs=output
    )

# Run app
if __name__ == "__main__":
    demo.launch()