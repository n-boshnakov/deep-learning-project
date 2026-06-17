from sklearn.metrics import accuracy_score, f1_score

def train_evaluate_pipeline(x_train, y_train, x_test, y_test, vectorizer, classifier):

    vectorized_x_train = vectorizer.fit_transform(x_train)
    vectorized_x_test = vectorizer.transform(x_test)

    model = classifier.fit(vectorized_x_train, y_train)

    predictions = classifier.predict(vectorized_x_test)

    test_accuracy = accuracy_score(y_test, predictions)
    test_macro_f1 = f1_score(y_test, predictions, average="macro")
    return model, test_accuracy, test_macro_f1