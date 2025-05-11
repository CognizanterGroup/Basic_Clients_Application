import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download
import os
from PIL import Image
import requests
from io import BytesIO

# ----------- Page Configuration -----------
st.set_page_config(page_title="Toxicity Text Classifier", page_icon="🛡️", layout="centered")

# ----------- Constants -----------
REPO_ID = "Datalictichub/Simple"
MODEL_FILENAME = "Comment_bert.pth"
TOKENIZER_NAME = "bert-base-uncased"
IMAGE_URL = "https://img.freepik.com/free-vector/stop-hate-speech-concept_23-2148584585.jpg"

# ----------- Download Required Resources -----------
def download_resources():
    with st.spinner("⏳ First-time setup: Downloading model resources..."):
        # Download tokenizer if not available locally
        tokenizer_dir = 'tokenizer/'
        st.success(f"✅ Tokenizer exist")
        if not os.path.exists(tokenizer_dir):
            try:
                tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
                tokenizer.save_pretrained(tokenizer_dir)
                st.success(f"✅ Tokenizer downloaded and saved to {tokenizer_dir}")
            except Exception as e:
                st.error(f"❌ Failed to download tokenizer: {str(e)}")
                return False

        # Download model
        try:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            st.success(f"✅ Model downloaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            return False

# ----------- Load Model and Tokenizer -----------
@st.cache_resource
def load_model():
    try:
        # Check if tokenizer directory exists, if not try to download it
        if not os.path.exists('tokenizer/'):
            if not download_resources():
                return None, None
        
        # Download the model file (will use cached version if already downloaded)
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        # Initialize the model and load weights
        model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=5)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        
        # Load the tokenizer from local directory or download if needed
        try:
            if os.path.exists('tokenizer/'):
                tokenizer = BertTokenizer.from_pretrained('tokenizer')
            else:
                tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
                # Save tokenizer locally for future use
                tokenizer.save_pretrained('tokenizer')
                
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            return None, None
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# ----------- Helper Function to Predict -----------
def predict(text, model, tokenizer):
    if model is None or tokenizer is None:
        return None
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    return probs

# ----------- Get Header Image -----------
@st.cache_data
def get_image():
    try:
        # Try to load local image first
        if os.path.exists("Hate speech.jpg"):
            return Image.open("Hate speech.jpg")
        # Otherwise load from URL and save locally for future use
        else:
            response = requests.get(IMAGE_URL)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                # Save image locally for future use
                try:
                    img.save("Hate speech.jpg")
                    st.success("✅ Header image downloaded and saved locally")
                except Exception as e:
                    st.warning(f"Could not save image locally: {str(e)}")
                return img
            else:
                st.warning(f"Could not download image: HTTP status {response.status_code}")
                return None
    except Exception as e:
        st.warning(f"Could not load image: {str(e)}")
        return None

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
# Start by ensuring resources are downloaded
if not os.path.exists('tokenizer') or not os.path.exists(os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub', 'models--Datalictichub--Simple', 'snapshots')):
    st.warning("🚀 First time running the app. Setting up resources...")
    success = download_resources()
    if not success:
        st.error("Failed to download required resources. Please check your internet connection and try again.")
        st.stop()

# Load model and tokenizer
model, tokenizer = load_model()

# Display a message if model loading is in progress
if model is None or tokenizer is None:
    st.error("Failed to load the model or tokenizer. Please refresh the page to try again.")
    st.stop()

# Display header image
st.info("Model and tokenizer loaded successfully! The app is ready to use.")
image = get_image()
if image:
    st.image(image, use_container_width=True)

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
    elif model is None or tokenizer is None:
        st.error("Model or tokenizer could not be loaded. Please check the console for errors.")
    else:
        with st.spinner("Analyzing..."):
            probabilities = predict(text_input, model, tokenizer)
            
        if probabilities is not None:
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
            top_prob = results_df.iloc[0]["Probability (%)"]
            st.success(f"**Predicted Class:** {top_class} ({top_prob}%)")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using BERT and Streamlit")
