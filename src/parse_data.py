import pandas as pd

def load_and_split_data(data_file):
    columns = [
    "id", "label", "statement", "subjects", "speaker", 
    "speaker_job", "state_info", "party_affiliation", 
    "barely_true_counts", "false_counts", "half_true_counts", 
    "mostly_true_counts", "pants_on_fire_counts", "context"
]
    loaded_data = pd.read_csv(data_file, sep="\t", header=None, names=columns, quoting=3)
    data_clean = loaded_data.dropna(subset=["label", "statement"])
    data_split = data_clean["statement"], data_clean["label"]

    return data_split