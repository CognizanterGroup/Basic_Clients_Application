import streamlit as st
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ----------- Page Configuration -----------
st.set_page_config(page_title="Toxicity Text Classifier", page_icon="🛡️", layout="centered")

# ----------- Constants -----------
MODEL_PATH = "comment_bert.pth"
MODEL_URL = "https://huggingface.co/Datalictichub/Simple/resolve/main/Comment_bert.pth"
TOKENIZER_PATH = "tokenizer/"

# ----------- Ensure Model File Exists -----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

# ----------- Load Model and Tokenizer -----------
@st.cache_resource
def load_model():
    download_model()

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_model()

# ----------- Helper Function to Predict -----------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return probs

# ----------- Static Information -----------
labels = ["Neutral", "Mildly Toxic", "Moderately Toxic", "Highly Toxic", "Extremely Toxic"]
descriptions = {
    "Neutral": "Non-toxic or very low toxicity.",
    "Mildly Toxic": "Slight signs of toxic language.",
    "Moderately Toxic": "Noticeable toxicity but not severe.",
    "Highly Toxic": "Strong toxic content.",
    "Extremely Toxic": "Very aggressive, harmful, or hate speech."
}

# ----------- App Layout -----------
st.image("Hate speech.jpg", use_container_width=True)
st.title("🛡️ Toxicity Text Classification App")
st.write("""
Welcome to the **Toxicity Classifier**!  
Enter any text below and let our BERT-powered model assess its toxicity level.  
We will show you the prediction probabilities and a clear visualization!
""")

text_input = st.text_area("Enter text to classify:", height=150)

if st.button("Classify Text"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        with st.spinner("Analyzing..."):
            probabilities = predict(text_input)

        results_df = pd.DataFrame({
            "Class": labels,
            "Description": [descriptions[label] for label in labels],
            "Probability (%)": (probabilities * 100).round(2)
        }).sort_values(by="Probability (%)", ascending=False)

        st.subheader("🔍 Prediction Results")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("📊 Probability Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Probability (%)"], y=results_df["Class"], palette="viridis", ax=ax)
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Class")
        ax.set_xlim(0, 100)
        st.pyplot(fig)

        top_class = results_df.iloc[0]["Class"]
        st.success(f"**Predicted Class:** {top_class}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using BERT and Streamlit")
