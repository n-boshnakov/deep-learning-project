# Fake News Detector (LIAR Dataset)

This project is a fake news classification system based on Natural Language Processing (NLP). The models are trained on the popular [LIAR Dataset](https://paperswithcode.com/dataset/liar), which contains short statements annotated with 6 degrees of truthfulness (ranging from *pants-on-fire* to *true*).

## Project Architecture
The project adheres to strict industry standards for structuring Python modules:

* `experiments/` - Execution scripts for different machine learning experiments (H01 to H05).
* `fake_news_detector/` - Core package containing the business logic (data parsing, vectorization, pipelines).
* `liar_dataset/` - Contains the liar dataset, split into training, testing and validation sets
* `notebooks/` - Jupyter Notebooks for Exploratory Data Analysis (EDA).
* `tests/` - Unit tests powered by `pytest` (Code coverage: 100%).

## Installation & Setup

1. Create a Virtual Environment:
   ```bash
   conda create -n deep_learning python=3.10
   conda activate deep_learning
   ```

1. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

1. Run a chosen experiment:

    ```bash
    python -m experiments.05_run_h05_glove_logreg
    ```

## Code Quality (Git Hooks)

The repository is protected by a pre-push hook that enforces code quality standards. To run type checking (mypy), linting/formatting (yapf, isort), and test coverage (coverage), execute:

    ```bash
    sh .git-hooks/pre-push
    ```