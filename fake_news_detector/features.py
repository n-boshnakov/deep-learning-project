import spacy
import numpy as np
import gensim.downloader as gensim

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
    
class GensimVectorizer:
    def __init__(self, model_name="glove-twitter-25"):
        self.model = gensim.load(model_name)
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        all_vectors = []

        for sentence in x:
            split_sentence = sentence.lower().split()
            vectorized_sentence = []
            for word in split_sentence:
                # avoid errors if the word is not present in the model
                if word in self.model:
                    vectorized_sentence.append(self.model[word])
            if len(vectorized_sentence) > 0:
                vectorized_sentence = np.mean(vectorized_sentence, axis=0)
            else:
                vectorized_sentence = np.zeros(self.model.vector_size)

            all_vectors.append(vectorized_sentence)
            
        return np.array(all_vectors)

    def fit_transform(self, x, y=None):
        return self.transform(x)