import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, f1_score

def train_evaluate_pipeline(
    x_train: pd.Series, 
    y_train: pd.Series, 
    x_test: pd.Series, 
    y_test: pd.Series, 
    vectorizer: BaseEstimator, 
    classifier: BaseEstimator
) -> tuple[BaseEstimator, float, float]:

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    model = classifier.fit(vectorized_x_train, y_train)

    predictions = classifier.predict(vectorized_x_test)

    test_macro_f1 = f1_score(y_test, predictions, average="macro")
    test_macro_precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    
    return model, test_macro_f1, test_macro_precision