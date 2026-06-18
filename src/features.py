import spacy
import numpy as np

class SpacyVectorizer:
    def __init__(self, model_name="en_core_web_md"):
        self.nlp = spacy.load(model_name)
    
    def fit(self, x, y=None):
        # spaCy is already trained, so it doesn't need to learn anything.
        return self
    
    def transform(self, x):
        all_vectors = []
        for sentence in x:
            doc = self.nlp(sentence)
            curr_vector = doc.vector
            all_vectors.append(curr_vector)
        return np.array(all_vectors)
    
    def fit_transform(self, x, y=None):
        return self.transform(x)