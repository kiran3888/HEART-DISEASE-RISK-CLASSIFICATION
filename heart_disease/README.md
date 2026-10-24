# Heart Disease Risk Classification (Educational ML Mini Project)

A ready-to-run Streamlit web app that trains a classifier on the UCI Heart Disease dataset, evaluates it on a held-out test set, and lets you try a sample record.

## Requirements
- Python 3.10+ recommended
- Internet connection for the first run so the official dataset can be fetched
- Windows, macOS, or Linux

## Quick start (Windows)
1. Extract the ZIP folder.
2. Open the extracted `heart_disease_risk_classifier` folder.
3. Double-click `run_windows.bat`.

Or use Command Prompt:
```bat
cd path\to\heart_disease_risk_classifier
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Dataset
The app fetches dataset ID 45 from the UCI Machine Learning Repository via `ucimlrepo` on first run, then caches a copy at `data/heart_disease.csv`.

Official source: https://archive.ics.uci.edu/dataset/45/heart+disease

If you prefer manual download:
1. Open the UCI page above and use its Download option, or install `ucimlrepo` and use the official import method.
2. Save a CSV with the feature columns `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal` and a target column named `num` (or `goal`, `target`, `diagnosis`) to `data/heart_disease.csv`.
3. Missing values are handled by the app's preprocessing pipeline.

## ML workflow
- Load data and handle missing values.
- Convert the original target into binary classes: `0` means no disease; target values `1–4` are combined as disease present.
- Split into training and test sets using a stratified split.
- Train Random Forest or Logistic Regression.
- Report accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC where available.
- Enter sample feature values to see the model output.

## Important limitations
- This is a small, historical dataset for learning, not clinical use.
- The displayed probability is the model's estimated probability for the positive class, not a calibrated medical risk.
- The app is not a diagnostic tool. Do not use it to diagnose, treat, or rule out heart disease.
- Model results depend on the split, dataset version, and environment.
