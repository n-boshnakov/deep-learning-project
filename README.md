# Fake News Detector (LIAR Dataset)

This project is an exploration on creating and improving a fake news classification system through NLP and deep learning techniques. The models are trained on the popular LIAR Dataset ([Hugging Face](https://paperswithcode.com/dataset/liar), [GitHub](https://github.com/tfs4/liar_dataset)), which contains short statements annotated with 6 degrees of truthfulness (ranging from *pants-on-fire* to *true*).

## Project Architecture

The project has the followig architecture:

* `experiments/` - Execution scripts for different machine learning experiments (H01 to H05).
* `fake_news_detector/` - Core package containing the business logic (data parsing, vectorization, pipelines).
* `liar_dataset/` - Contains the liar dataset, split into training, testing and validation sets.
* `notebooks/` - Jupyter Notebooks for Exploratory Data Analysis (EDA).
* `tests/` - Unit tests powered by `pytest` (Code coverage: 100%).

## Installation & Setup

1. Create a virtual environment:
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

1. Run `python -m experiments.<file-name`, for example:

    ```bash
    python -m experiments.04_run_h04_word2vec_logreg
    ```

## Accessing the Web UI

1. Run:
    ```bash
    streamlit run app.py
    ```

1. The application will become accessible on specified `Local` and `Network` URLs.

## Code Quality (Git Hooks)

The repository is protected by a pre-push hook that enforces code quality standards. To run type checking (mypy), linting/formatting (yapf, isort), and test coverage (coverage), run:

```bash
sh .git-hooks/pre-push
```

