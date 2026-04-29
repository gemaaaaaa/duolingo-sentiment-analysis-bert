import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Page configuration
st.set_page_config(
    page_title="Duolingo Review Sentiment Analysis",
    page_icon="🦉",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    [data-testid="stHeading"] h1 {
        white-space: nowrap !important;
        font-size: clamp(1.5rem, 4vw, 2.5rem) !important;
        text-align: center !important;
    }
    [data-testid="stAlert"] {
        text-align: center !important;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #ffd600, #ffff8d);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: #333;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 214, 0, 0.3);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff1744, #ff8a80);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3);
    }
    .confidence-bar {
        margin-top: 1rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the fine-tuned BERT model and tokenizer."""
    model_path = "checkpoint-33750"  # Current directory contains the checkpoint
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device

def predict_sentiment(text, tokenizer, model, device):
    """Predict sentiment for the given text."""
    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Get all probabilities
    probs = probabilities[0].cpu().numpy()
    
    return predicted_class, confidence, probs

def get_sentiment_label(class_id):
    """Map class ID to sentiment label."""
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels.get(class_id, "Unknown")

def get_sentiment_emoji(class_id):
    """Map class ID to emoji."""
    emojis = {0: "😞", 1: "😐", 2: "😊"}
    return emojis.get(class_id, "❓")

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Duolingo Review Sentiment Analysis")
    st.markdown("**Model ini berbasis arsitektur BERT yang dikembangkan dengan dataset Duolingo App User Review Play Store 2025**")
    
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model... (this might take a moment)"):
        try:
            tokenizer, model, device = load_model()
            st.success(f"✅ Model loaded successfully! (Device: {device})")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return
    
    st.markdown("---")
    
    # Initialize session state for review text
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    
    # Input section
    st.subheader("Input Teks Ulasan")
    
    # Example reviews
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("😊 Positive Example", use_container_width=True):
            st.session_state.review_text = "This app is amazing! It really helps me learn a new language in a fun and interactive way. Highly recommended!"
            st.rerun()
    
    with col2:
        if st.button("😐 Neutral Example", use_container_width=True):
            st.session_state.review_text = "The app is okay. Some features are good but there are also things that need improvement."
            st.rerun()
    
    with col3:
        if st.button("😞 Negative Example", use_container_width=True):
            st.session_state.review_text = "The new AI features ruined the app. It feels less personal and more robotic now. I miss the old Duolingo experience."
            st.rerun()
    
    st.markdown("")
    
    # Text input with session state
    review_text = st.text_area(
        "Atau ketik ulasan kamu di sini :",
        value=st.session_state.review_text,
        height=150,
        placeholder="This app really helps me learn English in a fun way!"
    )
    
    if st.button("Analisis Sentimen", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("⚠️ Silakan masukkan review terlebih dahulu!")
        else:
            with st.spinner("Menganalisis sentimen..."):
                # Get prediction
                st.markdown("---")
                predicted_class, confidence, probs = predict_sentiment(
                    review_text, tokenizer, model, device
                )
                
                sentiment_label = get_sentiment_label(predicted_class)
                sentiment_emoji = get_sentiment_emoji(predicted_class)
                                
                # Sentiment card
                sentiment_class = f"sentiment-{sentiment_label.lower()}"
                st.markdown(
                    f'<div class="{sentiment_class}">{sentiment_emoji} {sentiment_label}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("")
                
                # Confidence scores
                st.markdown("#### Confidence Scores:")
                
                # Progress bars for each sentiment
                labels = ["Negative", "Neutral", "Positive"]
                colors = ["red", "yellow", "green"]
                
                for i, (label, prob) in enumerate(zip(labels, probs)):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{label}**")
                    with col2:
                        st.progress(float(prob), text=f"{prob*100:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p><a href="https://huggingface.co/google-bert/bert-base-multilingual-cased" target="_blank">BERT Model</a></p>
            <p><a href="https://www.kaggle.com/datasets/belalakhter/duolingo-app-user-review-play-store-dataset-2025" target="_blank">Duolingo App User Review Play Store Dataset 2025</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
