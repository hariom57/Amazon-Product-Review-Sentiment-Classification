# Amazon Product Review Sentiment Classification

A modular machine learning pipeline to perform sentiment analysis on large-scale Amazon product reviews. This project focuses on end-to-end data processing, feature engineering, and model evaluation.

## 🚀 Project Overview
- **Objective:** Automate sentiment categorization (Positive/Negative) of product reviews.
- **Methodology:** Implemented a TF-IDF vectorization pipeline coupled with a Logistic Regression classifier, optimized for interpretability and performance.
- **Key Skills:** Python, NLP, scikit-learn, Model Pipeline Architecture.

## 🛠 Tech Stack
- **Languages:** Python
- **Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- **Environment:** Designed with modularity for future integration of Transformer architectures (BERT/DistilBERT).

## 📂 Project Structure
```text
├── data/               # (Git-ignored) Raw/Processed datasets(I'm not using it because I'm using load_dataset from Hugging Face)
├── src/                # Modular source code
│   ├── preprocess.py   # Data cleaning and vectorization
│   ├── train.py        # Model training and orchestration
│   └── evaluate.py     # Evaluation metrics and reporting
├── requirements.txt    # Project dependencies
└── README.md
```

### How to Reproduce


1. Clone this repo: `git clone <url>`
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate
```


3. Install dependencies: `pip install -r requirements.txt`
4. Run the pipeline: `python src/train.py`