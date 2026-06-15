import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure that the charts directory exists
os.makedirs("charts", exist_ok=True)

# 1. Define the original 14 columns of the LIAR dataset
columns = [
    "id", "label", "statement", "subjects", "speaker", 
    "speaker_job", "state_info", "party_affiliation", 
    "barely_true_counts", "false_counts", "half_true_counts", 
    "mostly_true_counts", "pants_on_fire_counts", "context"
]

print("⏳ Starting comprehensive Exploratory Data Analysis (EDA)...")
train_df = pd.read_csv("liar_dataset/train.tsv", sep="\t", header=None, names=columns, quoting=3)
test_df = pd.read_csv("liar_dataset/test.tsv", sep="\t", header=None, names=columns, quoting=3)

print("\n==================================================")
print("1. OBSERVATIONS COUNT")
print("==================================================")
print(f"Total number of rows in the training set (Train): {len(train_df)}")
print(f"Total number of rows in the test set (Test):       {len(test_df)}")

print("\n==================================================")
print("2. FEATURES COUNT & DATA TYPES")
print("==================================================")
print(f"Total number of features (columns): {len(train_df.columns)}")
print("\nData types by column:")
print(train_df.dtypes)

print("\n==================================================")
print("3. STATISTICAL DEPENDENCIES & DISTRIBUTION")
print("==================================================")
# Sort classes by logical truth scale
class_order = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

print("Distribution of the 6 classes in the training data (absolute count and %):")
class_counts = train_df["label"].value_counts().reindex(class_order)
for cls in class_order:
    count = class_counts[cls]
    percentage = (count / len(train_df)) * 100
    print(f" - {cls:<12}: {count:<5} examples ({percentage:.2f}%)")

# Visualization 1: Distribution of the 6 classes
plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x="label", order=class_order, hue="label", palette="vlag", edgecolor="black", legend=False)
plt.title("Frequency Distribution of the 6 Truth Levels in LIAR (Train)")
plt.xlabel("Truth Class")
plt.ylabel("Number of Statements")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("charts/eda_class_distribution.png")
plt.close()

print("\n==================================================")
print("4. ANOMALY & MISSING VALUES ANALYSIS")
print("==================================================")
print("Number of missing values (NaN) by column in the training set:")
null_summary = train_df.isnull().sum()
print(null_summary[null_summary > 0])

# Calculate text length (word count) in statements for anomaly detection
train_df["word_count"] = train_df["statement"].astype(str).apply(lambda x: len(x.split()))

print("\nStatement length statistics (word count):")
print(train_df["word_count"].describe())

# Find potential text anomalies (excessively short or long texts)
very_short = train_df[train_df["word_count"] <= 2]
very_long = train_df[train_df["word_count"] > 50]              
print(f"\nLength anomalies:")
print(f" - Statements with 2 or fewer words: {len(very_short)} examples")
print(f" - Statements with more than 50 words:         {len(very_long)} examples")

# Visualization 2: Histogram of statement lengths
plt.figure(figsize=(10, 5))
sns.histplot(train_df["word_count"], bins=40, color="purple", kde=True, edgecolor="black")
plt.title("Distribution of Statement Lengths (Word Count)")
plt.xlabel("Number of words in a single statement")
plt.ylabel("Frequency")
plt.xlim(0, 80)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("charts/eda_statement_lengths.png")
plt.close()

print("\n✅ Analysis complete! Charts have been successfully saved in the 'charts/' folder.")