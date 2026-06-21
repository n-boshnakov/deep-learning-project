import os

import joblib
import streamlit as st

from fake_news_detector.features import GensimVectorizer, SpacyVectorizer


@st.cache_resource
def load_nlp_pipeline():
    model_path = "models/word2vec_lg_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def main():
    # Title and subheader
    st.title("Fake News Detector")
    st.subheader("NLP Text Classification System")

    # Brief description of the project, dataset, and classes
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

    # Load the model pipeline
    pipeline = load_nlp_pipeline()

    if pipeline is None:
        st.error(
            "Model file not found. Please run the training script to generate it."
        )
        return

    vectorizer = pipeline["vectorizer"]
    classifier = pipeline["classifier"]

    # Dynamically display the current vectorizer and classifier names
    vec_name = type(vectorizer).__name__
    clf_name = type(classifier).__name__

    st.write(f"**Active Model:** {vec_name} + {clf_name}")
    st.write("---")

    # Text area for user input
    user_input = st.text_area("Enter text for classification:")

    # Classification button
    if st.button("Classify"):
        if not user_input.strip():
            st.warning(
                "Please enter some text before starting the classification.")
        else:
            # Process the text and make a prediction
            vectorized_text = vectorizer.transform([user_input])
            prediction = classifier.predict(vectorized_text)[0]

            # Display the result
            st.write(f"**Classification:** {prediction}")


if __name__ == "__main__":
    main()
