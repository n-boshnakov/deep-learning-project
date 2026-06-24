import os
import pickle

import joblib
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

from fake_news_detector.modeling import (BaselineEmbeddingNet, HybridBertFakeNewsNet,
                                         HybridRNNFakeNewsNet)

# Supported active models: "h04_sklearn", "h06_baseline", "h08_hybrid_gru", "h14_roberta"
ACTIVE_MODEL_TYPE = "h14_roberta"

MAX_SEQ_LEN = 50
EMBEDDING_DIM = 50
RNN_HIDDEN_DIM = 64
H14_MODEL_NAME = "roberta-base"
H14_MAX_SEQ_LEN = 64

IDX_TO_LABEL = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}


@st.cache_resource
def load_h04_pipeline():
    model_path = "models/word2vec_lg_pipeline.pkl"
    if not os.path.exists(model_path):
        return None, None
    pipeline_data = joblib.load(model_path)
    return pipeline_data.get("vectorizer"), pipeline_data.get("classifier")


@st.cache_resource
def load_h06_pipeline():
    weights_path = "models/h06_pytorch_embeddings_weights.pth"
    vocab_path = "models/h06_pytorch_embeddings_word2idx.pkl"

    if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
        return None, None

    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    model = BaselineEmbeddingNet(vocab_size=len(word2idx),
                                 embed_dim=EMBEDDING_DIM,
                                 num_classes=6)
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, word2idx


@st.cache_resource
def load_h08_pipeline():
    weights_path = "models/h08_hybrid_gru_weights.pth"
    vocab_path = "models/h08_gru_word2idx.pkl"
    prep_path = "models/metadata_preprocessor.pkl"

    if not os.path.exists(weights_path) or not os.path.exists(
            vocab_path) or not os.path.exists(prep_path):
        return None, None, None

    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)

    preprocessor = joblib.load(prep_path)

    dummy_df = pd.DataFrame([{
        "barely_true_counts": 0.0,
        "false_counts": 0.0,
        "half_true_counts": 0.0,
        "mostly_true_counts": 0.0,
        "pants_on_fire_counts": 0.0,
        "party_affiliation": "unknown",
        "state_info": "unknown",
        "speaker_job": "unknown",
        "speaker": "unknown",
        "context": "unknown",
        "subjects": "unknown"
    }])
    dummy_tensor = preprocessor.transform(dummy_df)
    meta_feature_count = dummy_tensor.shape[1]

    model = HybridRNNFakeNewsNet(vocab_size=len(word2idx),
                                 embed_dim=EMBEDDING_DIM,
                                 hidden_dim=RNN_HIDDEN_DIM,
                                 meta_dim=meta_feature_count,
                                 num_classes=6,
                                 rnn_type="GRU")
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, word2idx, preprocessor


@st.cache_resource
def load_h14_pipeline():
    weights_path = "models/h14_roberta_weights.pth"
    prep_path = "models/h14_metadata_preprocessor.pkl"

    if not os.path.exists(weights_path) or not os.path.exists(prep_path):
        return None, None, None

    preprocessor = joblib.load(prep_path)

    dummy_df = pd.DataFrame([{
        "barely_true_counts": 0.0,
        "false_counts": 0.0,
        "half_true_counts": 0.0,
        "mostly_true_counts": 0.0,
        "pants_on_fire_counts": 0.0,
        "party_affiliation": "unknown",
        "state_info": "unknown",
        "speaker_job": "unknown",
        "speaker": "unknown",
        "context": "unknown",
        "subjects": "unknown"
    }])
    dummy_tensor = preprocessor.transform(dummy_df)
    meta_feature_count = dummy_tensor.shape[1]

    model = HybridBertFakeNewsNet(bert_model_name=H14_MODEL_NAME,
                                  meta_dim=meta_feature_count)
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(H14_MODEL_NAME)

    return model, preprocessor, tokenizer


def preprocess_text_pytorch(text: str, word2idx: dict) -> torch.Tensor:
    tokens = [word2idx.get(word.lower(), 1) for word in text.split()]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens = tokens + [0] * (MAX_SEQ_LEN - len(tokens))
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def main():
    st.set_page_config(page_title="Fake News Detector",
                       page_icon="🕵️‍♂️",
                       layout="centered")

    st.title("Fake News Detector")
    st.subheader("Text Classification System")

    st.write("""
    This project uses machine learning algorithms to predict the credibility of a given statement. 
    The system is trained on the popular **LIAR** dataset, which categorizes short statements into 6 degrees of truthfulness.
    """)

    if ACTIVE_MODEL_TYPE == "h04_sklearn":
        sk_vectorizer, sk_classifier = load_h04_pipeline()
        if sk_classifier is None:
            st.error(
                "H04 Scikit-Learn files not found in the `models/` directory.")
            return
        st.write(
            f"**Active Model:** {type(sk_classifier).__name__} (H04 - Classical ML)"
        )

    elif ACTIVE_MODEL_TYPE == "h06_baseline":
        pt_model, pt_vocab = load_h06_pipeline()
        if pt_model is None:
            st.error("H06 PyTorch files not found in the `models/` directory.")
            return
        st.write(
            f"**Active Model:** {type(pt_model).__name__} (H06 - Deep Learning Baseline)"
        )

    elif ACTIVE_MODEL_TYPE == "h08_hybrid_gru":
        pt_model, pt_vocab, preprocessor = load_h08_pipeline()
        if pt_model is None:
            st.error(
                "H08 PyTorch Hybrid files (including preprocessor) not found in `models/`."
            )
            return
        st.write(
            f"**Active Model:** {type(pt_model).__name__} (H08 - Hybrid text + metadata)"
        )

    elif ACTIVE_MODEL_TYPE == "h14_roberta":
        pt_model, preprocessor, tokenizer = load_h14_pipeline()
        if pt_model is None:
            st.error(
                "H14 RoBERTa files not found in `models/`. Expected: h14_roberta_weights.pth, h14_metadata_preprocessor.pkl"
            )
            return
        st.write(
            f"**Active Model:** {type(pt_model).__name__} (H14 - Hybrid RoBERTa + metadata)"
        )

    else:
        st.error("Invalid ACTIVE_MODEL_TYPE specified in the code.")
        return

    st.write("---")

    user_input = st.text_area("Enter statement for classification:",
                              height=100)

    meta_df = None
    if ACTIVE_MODEL_TYPE in ("h08_hybrid_gru", "h14_roberta"):
        st.markdown("#### Additional Context (Metadata)")
        st.markdown(
            "This model requires context about the speaker to make an accurate prediction."
        )

        col1, col2 = st.columns(2)
        with col1:
            speaker = st.text_input("Speaker Name", value="donald-trump")
            party = st.text_input("Party Affiliation", value="republican")
            state = st.text_input("State", value="New York")
            job = st.text_input("Speaker Job", value="President-Elect")
        with col2:
            subjects = st.text_input("Subject(s)", value="economy, jobs")
            context = st.text_input("Context (e.g. debate, tweet)",
                                    value="tweet")

        st.markdown("**Historical Lie Count (Optional / Known History)**")
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        barely = col_c1.number_input("Barely True", min_value=0, value=0)
        false_cnt = col_c2.number_input("False", min_value=0, value=0)
        half = col_c3.number_input("Half True", min_value=0, value=0)
        mostly = col_c4.number_input("Mostly True", min_value=0, value=0)
        pants = col_c5.number_input("Pants on Fire", min_value=0, value=0)

        meta_df = pd.DataFrame([{
            "barely_true_counts": float(barely),
            "false_counts": float(false_cnt),
            "half_true_counts": float(half),
            "mostly_true_counts": float(mostly),
            "pants_on_fire_counts": float(pants),
            "party_affiliation": party.lower(),
            "state_info": state.lower(),
            "speaker_job": job.lower(),
            "speaker": speaker.lower(),
            "context": context.lower(),
            "subjects": subjects.lower()
        }])

    st.write("---")

    if st.button("Classify Statement",
                 type="primary",
                 use_container_width=True):
        if not user_input.strip():
            st.warning(
                "Please enter a statement before starting the classification.")
            return

        with st.spinner('Analyzing text and data...'):
            try:
                if ACTIVE_MODEL_TYPE == "h04_sklearn":
                    vectorized_input = sk_vectorizer.transform([user_input])
                    prediction_label = sk_classifier.predict(
                        vectorized_input)[0]

                elif ACTIVE_MODEL_TYPE == "h06_baseline":
                    input_tensor = preprocess_text_pytorch(
                        user_input, pt_vocab)
                    with torch.no_grad():
                        output = pt_model(input_tensor)
                        prediction_idx = torch.argmax(output, dim=1).item()
                    prediction_label = IDX_TO_LABEL.get(
                        prediction_idx, "Unknown")

                elif ACTIVE_MODEL_TYPE == "h08_hybrid_gru":
                    text_tensor = preprocess_text_pytorch(user_input, pt_vocab)
                    meta_tensor = preprocessor.transform(meta_df)

                    with torch.no_grad():
                        output = pt_model(text_tensor, meta_tensor)
                        prediction_idx = torch.argmax(output, dim=1).item()
                    prediction_label = IDX_TO_LABEL.get(
                        prediction_idx, "Unknown")

                elif ACTIVE_MODEL_TYPE == "h14_roberta":
                    encoding = tokenizer(user_input,
                                         max_length=H14_MAX_SEQ_LEN,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                    meta_tensor = preprocessor.transform(meta_df)

                    with torch.no_grad():
                        output = pt_model(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            meta_input=meta_tensor)
                        prediction_idx = torch.argmax(output, dim=1).item()
                    prediction_label = IDX_TO_LABEL.get(
                        prediction_idx, "Unknown")

                st.subheader("Classification Result")
                st.info(f"**Prediction:** `{prediction_label}`")

            except Exception as e:
                st.error(f"An error occurred during classification: {e}")


if __name__ == "__main__":
    main()
