# Fake News Detector (LIAR Dataset)

This project is an exploration on creating and improving a fake news classification system through NLP and deep learning techniques. The models are trained on the popular LIAR Dataset ([Hugging Face](https://paperswithcode.com/dataset/liar), [GitHub](https://github.com/tfs4/liar_dataset)), which contains short statements annotated with 6 degrees of truthfulness (ranging from *pants-on-fire* to *true*).

## Project Architecture

The project has the following architecture:

* `experiments/` - Execution scripts for different machine learning experiments (H01 to H15).
* `fake_news_detector/` - Core package containing the business logic (data parsing, vectorization, pipelines, various utility functions).
* `liar_dataset/` - Contains the liar dataset, split into training, testing and validation sets.
* `notebooks/` - Jupyter Notebooks for Exploratory Data Analysis (EDA).
* `tests/` - Unit tests powered by `pytest` (Code coverage: 100%).

## Installation & Setup

1. Create a virtual environment (Windows only):
   ```bash
   conda create -n deep_learning python=3.10
   conda activate deep_learning
   ```

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running Experiments

To run a chosen experiment from `/experiments`:

1. Run `python -m experiments.<file-name>`, for example:

    ```bash
    python -m experiments.04_run_h04_word2vec_logreg
    ```

## Model Files

Trained weights are not committed to the repository. After running an experiment, place the output files in the `models/` directory. The web app expects the following files depending on the active model:

| `ACTIVE_MODEL_TYPE` | Required files |
|---|---|
| `h04_sklearn` | `models/word2vec_lg_pipeline.pkl` |
| `h06_baseline` | `models/h06_pytorch_embeddings_weights.pth`, `models/h06_pytorch_embeddings_word2idx.pkl` |
| `h08_hybrid_gru` | `models/h08_hybrid_gru_weights.pth`, `models/h08_gru_word2idx.pkl`, `models/metadata_preprocessor.pkl` |
| `h14_roberta` | `models/h14_roberta_weights.pth`, `models/h14_metadata_preprocessor.pkl` |

## Accessing the Web UI

1. Run an experiment of your choice.

1. Change the `ACTIVE_MODEL_TYPE` constant at the top of `app.py` to the value in the table above corresponding to the experiment you've run.

> [!NOTE]
> The app will display an error and exit if any required file is missing.

1. Start the streamlit app:
    ```bash
    streamlit run app.py
    ```

1. The application will become accessible on specified `Local` and `Network` URLs.

## Code Quality (Git Hooks)

The repository is protected by a pre-push hook that enforces code quality standards. To run type checking (mypy), linting/formatting (yapf, isort), and test coverage (coverage), run:

```bash
sh .git-hooks/pre-push
```

