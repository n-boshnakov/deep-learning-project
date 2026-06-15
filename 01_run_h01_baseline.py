import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

columns = [
    "id", "label", "statement", "subjects", "speaker", 
    "speaker_job", "state_info", "party_affiliation", 
    "barely_true_counts", "false_counts", "half_true_counts", 
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

train_df = pd.read_csv("liar_dataset/train.tsv", sep="\t", header=None, names=columns, quoting=3)
test_df = pd.read_csv("liar_dataset/test.tsv", sep="\t", header=None, names=columns, quoting=3)

# Cleanining up the data from rows missing crucial values
train_df = train_df.dropna(subset=["label", "statement"])
test_df = test_df.dropna(subset=["label", "statement"])

X_train, y_train = train_df["statement"], train_df["label"]
X_test, y_test = test_df["statement"], test_df["label"]

# Initializing and "training" the DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Generating predictions
baseline_predictions = dummy_clf.predict(X_test)

# Collecting test metrics
test_accuracy = accuracy_score(y_test, baseline_predictions)
test_macro_f1 = f1_score(y_test, baseline_predictions, average="macro")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1 * 100:.2f}%")
