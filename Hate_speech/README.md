Project Overview
End-to-end ML pipeline for classifying text (e.g., tweets) as hate speech, offensive language, or neither. Includes EDA, preprocessing, feature engineering (TF-IDF, PCA), modeling (Logistic Regression, Random Forest), hyperparameter tuning, evaluation, and model deployment.
Dataset: ~25K labeled tweets.
Key Technologies

Python 3.12+
Libraries: Pandas, NumPy, NLTK, Scikit-learn, Matplotlib, Seaborn, Joblib
Concepts: Text classification, vectorization, dimensionality reduction, GridSearchCV, multi-class metrics (F1, ROC-AUC)

Installation

Clone repo: git clone https://github.com/Goodluck96/hate-speech-detection.git
Setup venv: python -m venv venv && source venv/bin/activate
Install: pip install -r requirements.txt
NLTK downloads: Run notebook or nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4'])

Usage

Run Notebook(240266783).ipynb in Jupyter.
Train/evaluate models; best model saved to /model/.
Predict: Load model and preprocess new text (see notebook).

Results

Best Model: Random Forest (F1 ~0.80, ROC-AUC ~0.94)
Metrics table, confusion matrix, ROC/PR curves in /report/.
