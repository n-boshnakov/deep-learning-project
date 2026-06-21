import os
import pickle
import torch
import joblib
import streamlit as st

from fake_news_detector.modeling import BaselineEmbeddingNet

# --- CONFIGURATION ---
# To switch the active model, change this variable to "pytorch" or "sklearn"
ACTIVE_MODEL_TYPE = "pytorch"

MAX_SEQ_LEN = 50
EMBEDDING_DIM = 50

IDX_TO_LABEL = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

@st.cache_resource
def load_pytorch_pipeline():
    weights_path = "models/pytorch_weights.pth"
    vocab_path = "models/word2idx.pkl"
    
    if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
        return None, None

    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    model = BaselineEmbeddingNet(vocab_size=len(word2idx), embed_dim=EMBEDDING_DIM, num_classes=6)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))) 
    model.eval() 

    return model, word2idx

@st.cache_resource
def load_sklearn_pipeline(file_name="word2vec_lg_pipeline.pkl"):
    model_path = os.path.join("models", file_name)
    
    if not os.path.exists(model_path):
        return None, None

    pipeline_data = joblib.load(model_path)
    return pipeline_data.get("vectorizer"), pipeline_data.get("classifier")

def preprocess_text_pytorch(text: str, word2idx: dict) -> torch.Tensor:
    tokens = [word2idx.get(word.lower(), 1) for word in text.split()]
    
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens = tokens + [0] * (MAX_SEQ_LEN - len(tokens))
        
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️", layout="centered")
    
    st.title("Fake News Detector")
    st.subheader("Text Classification System")

    st.write("""
    This project uses machine learning algorithms to predict the credibility of a given statement. 
    The system is trained on the popular **LIAR** dataset, which categorizes short statements into 6 degrees of truthfulness:
    * `pants-fire`
    * `false`
    * `barely-true`
    * `half-true`
    * `mostly-true`
    * `true`
    """)

    # Load the specific model based on the hardcoded configuration
    if ACTIVE_MODEL_TYPE == "pytorch":
        pt_model, pt_vocab = load_pytorch_pipeline()
        if pt_model is None:
            st.error("PyTorch files not found in the `models/` directory.")
            return
        st.write(f"**Active Model:** {type(pt_model).__name__} (Deep Learning)")
        
    elif ACTIVE_MODEL_TYPE == "sklearn":
        sk_vectorizer, sk_classifier = load_sklearn_pipeline("word2vec_lg_pipeline.pkl")
        if sk_classifier is None:
            st.error("Scikit-Learn file not found in the `models/` directory.")
            return
        st.write(f"**Active Model:** {type(sk_classifier).__name__} (Classical ML)")
    
    else:
        st.error("Invalid ACTIVE_MODEL_TYPE specified in the code.")
        return

    st.write("---")

    user_input = st.text_area("Enter text for classification:", height=150)

    if st.button("Classify", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text before starting the classification.")
            return
            
        with st.spinner('Analyzing text...'):
            try:
                if ACTIVE_MODEL_TYPE == "pytorch":
                    input_tensor = preprocess_text_pytorch(user_input, pt_vocab)
                    with torch.no_grad():
                        output = pt_model(input_tensor)
                        prediction_idx = torch.argmax(output, dim=1).item()
                    prediction_label = IDX_TO_LABEL.get(prediction_idx, "Unknown")
                
                elif ACTIVE_MODEL_TYPE == "sklearn":
                    vectorized_input = sk_vectorizer.transform([user_input])
                    prediction_label = sk_classifier.predict(vectorized_input)[0]

                st.subheader("Result")
                st.info(f"**Prediction:** `{prediction_label}`")
                
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")

if __name__ == "__main__":
    main()