# Rumor Detection with BERT + BiLSTM

This project implements a BERT + BiLSTM model to classify whether a piece of text is a **rumor or non-rumor**. The model is trained on the [PHEME Dataset for Rumour Detection](https://www.kaggle.com/datasets/nicolemichelle/pheme-dataset-for-rumour-detection).

---

## Dataset

- Source: [Kaggle - PHEME Dataset for Rumour Detection](https://www.kaggle.com/datasets/nicolemichelle/pheme-dataset-for-rumour-detection)
- Includes multiple Twitter conversations labeled as `rumor` or `non-rumor`.
- Only the top-level tweet (`text`) and its label (`is_rumor`) are used.

The file is saved locally as: rumour_dataset.csv

---

## Model Overview

BERT (bert-base-uncased) for language embeddings.

Bidirectional LSTM for sequence modeling.

Dense layers for classification.

Final output: Rumor (1) or Non-Rumor (0).

---

## Running the Project
1. Clone the repo and place the dataset CSV in the root directory.

2. Install dependencies:

``` 
pip install -r requirements.txt 
```

3. Run the model:

```
python rumor_detection.py
```

The script trains a model, evaluates it on a test set, and prints classification metrics.


