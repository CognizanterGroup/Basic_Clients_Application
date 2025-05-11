import streamlit as st
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO
import time

# ----------- Page Configuration -----------
st.set_page_config(page_title="Toxicity Text Classifier", page_icon="🛡️", layout="wide")

# ----------- Constants -----------
REPO_ID = "Datalictichub/Simple"
MODEL_FILENAME = "Comment_bert.pth"
TOKENIZER_NAME = "bert-base-uncased"
IMAGE_URL = "https://img.freepik.com/free-vector/stop-hate-speech-concept_23-2148584585.jpg"

# ----------- Sidebar Navigation -----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Classifier App"])

# ----------- Helper Functions -----------
def download_model_with_progress():
    """Download model with progress bar"""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We'll implement a manual progress simulation since huggingface_hub doesn't
        # provide direct progress callback in a way we can easily use with Streamlit
        status_text.text("Starting download...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Simulate progress while doing the actual download
        for i in range(10, 90, 10):
            progress_bar.progress(i)
            status_text.text(f"Downloading: {i}% complete")
            time.sleep(0.5)
            
        # Download the model file
        model_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=MODEL_FILENAME,
            force_download=True,  # Force download even if cached
            resume_download=True,  # Allow resuming partial downloads
        )
        
        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("Download complete!")
        time.sleep(1)  # Give user time to see the completion
        
        return True, model_path
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return False, None

def download_tokenizer():
    """Download and save tokenizer"""
    try:
        tokenizer_dir = 'tokenizer/'
        if not os.path.exists(tokenizer_dir):
            with st.spinner("Downloading tokenizer..."):
                tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
                tokenizer.save_pretrained(tokenizer_dir)
                st.success(f"✅ Tokenizer downloaded and saved")
        else:
            st.success(f"✅ Tokenizer already exists")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download tokenizer: {str(e)}")
        return False

def get_image():
    """Get header image from local storage or download it"""
    try:
        # Try to load local image first
        if os.path.exists("Hate speech.jpg"):
            return Image.open("Hate speech.jpg")
        # Otherwise load from URL and save locally for future use
        else:
            with st.spinner("Downloading header image..."):
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

@st.cache_resource
def load_model():
    """Load the model and tokenizer"""
    try:
        # Check if model exists
        model_path = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 
                                 'hub', 'models--Datalictichub--Simple', 'snapshots')
        
        # Check subdirectories for the model file
        model_exists = False
        for root, dirs, files in os.walk(model_path):
            if MODEL_FILENAME in files:
                model_exists = True
                model_file_path = os.path.join(root, MODEL_FILENAME)
                break
        
        if not model_exists:
            model_file_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        
        # Load tokenizer
        if os.path.exists('tokenizer/'):
            tokenizer = BertTokenizer.from_pretrained('tokenizer/')
        else:
            tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
            tokenizer.save_pretrained('tokenizer/')
        
        # Initialize model and load weights
        model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=5)
        model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict(text, model, tokenizer):
    """Predict toxicity levels for input text"""
    if model is None or tokenizer is None:
        return None
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():  # No need to track gradients for inference
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

# ----------- Home Page -----------
def show_home_page():
    st.title("🛡️ Toxicity Text Classifier - Setup")
    
    # Show header image
    image = get_image()
    if image:
        st.image(image, use_container_width=True)
    
    st.markdown("""
    ## Welcome to the Toxicity Classifier Setup Page!
    
    Before using the app, you need to download the required model and resources.
    
    ### About the App
    This application uses a BERT-based model to classify text according to its toxicity level. 
    The model can identify five levels of toxicity from neutral to extremely toxic content.
    
    ### Setup Instructions
    1. Click the **Download Tokenizer** button to download the BERT tokenizer
    2. Click the **Download Model** button to download the classification model
    3. Once both downloads are complete, navigate to the **Classifier App** page in the sidebar
    """)
    
    # Check and download tokenizer
    tokenizer_exists = os.path.exists('tokenizer/')
    if tokenizer_exists:
        st.success("✅ Tokenizer is already downloaded")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download Tokenizer" if not tokenizer_exists else "Re-Download Tokenizer"):
            download_tokenizer()
    
    # Check and download model
    model_path = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 
                             'hub', 'models--Datalictichub--Simple', 'snapshots')
    
    model_exists = False
    if os.path.exists(model_path):
        for root, dirs, files in os.walk(model_path):
            if MODEL_FILENAME in files:
                model_exists = True
                break
    
    if model_exists:
        st.success("✅ Model is already downloaded")
    
    with col2:
        if st.button("Download Model" if not model_exists else "Re-Download Model"):
            with st.spinner("Preparing download..."):
                success, _ = download_model_with_progress()
                if success:
                    st.success("✅ Model downloaded successfully!")
                    st.info("You can now go to the Classifier App page in the sidebar")

# ----------- App Page -----------
def show_app_page():
    # Check if resources exist
    tokenizer_exists = os.path.exists('tokenizer/')
    
    model_path = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 
                             'hub', 'models--Datalictichub--Simple', 'snapshots')
    
    model_exists = False
    if os.path.exists(model_path):
        for root, dirs, files in os.walk(model_path):
            if MODEL_FILENAME in files:
                model_exists = True
                break
    
    # If resources don't exist, prompt to go to home page
    if not tokenizer_exists or not model_exists:
        st.warning("⚠️ Required resources not found. Please go to the Home page to download them first.")
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model or tokenizer. Please go to the Home page and try redownloading.")
        return
    
    # Top Image and Title
    image = get_image()
    if image:
        st.image(image, use_container_width=True)
    
    st.title("🛡️ Toxicity Text Classification App")
    st.write("""
    Welcome to the **Toxicity Classifier**!  
    Enter any text below and let our BERT-powered model assess its toxicity level.  
    We will show you the prediction probabilities and a clear visualization!
    """)
    
    # User Input
    text_input = st.text_area("Enter text to classify:", height=150)
    
    if st.button("Classify Text"):
        if text_input.strip() == "":
            st.warning("⚠️ Please enter some text to classify.")
        else:
            with st.spinner("Analyzing..."):
                probabilities = predict(text_input, model, tokenizer)
                
            if probabilities is not None:
                # Create DataFrame for table
                results_df = pd.DataFrame({
                    "Class": labels,
                    "Description": [descriptions[label] for label in labels],
                    "Probability (%)": (probabilities * 100).round(2)
                }).sort_values(by="Probability (%)", ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🔍 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Display Top Prediction
                    top_class = results_df.iloc[0]["Class"]
                    top_prob = results_df.iloc[0]["Probability (%)"]
                    st.success(f"**Predicted Class:** {top_class} ({top_prob}%)")
                
                with col2:
                    # Bar Plot
                    st.subheader("📊 Probability Distribution")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=results_df["Probability (%)"], y=results_df["Class"], palette="viridis", ax=ax)
                    ax.set_xlabel("Probability (%)")
                    ax.set_ylabel("Class")
                    ax.set_xlim(0, 100)
                    st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using BERT and Streamlit")

# ----------- Run the Selected Page -----------
if page == "Home":
    show_home_page()
else:
    show_app_page()
