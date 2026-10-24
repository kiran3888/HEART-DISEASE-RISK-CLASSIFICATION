import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

st.set_page_config(page_title="Heart Disease Risk Classifier", page_icon="❤️", layout="wide")
DATA_PATH = os.path.join("data", "heart_disease.csv")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Download the official UCI dataset through its Python package.
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=45)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    # Standardize column names and tolerate UCI's target column naming.
    df.columns = [str(c).strip().lower() for c in df.columns]
    target_col = next((c for c in ["num", "goal", "target", "diagnosis"] if c in df.columns), None)
    if target_col is None:
        raise ValueError("Could not find target column. Expected one of: num, goal, target, diagnosis.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing expected columns: " + ", ".join(missing))
    df = df[FEATURES + [target_col]].copy()
    for col in FEATURES + [target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target_col])
    # UCI target: 0 = no disease; 1–4 = disease present.
    df["target_binary"] = (df[target_col] > 0).astype(int)
    return df

def build_model(model_name, X_train):
    categorical = [c for c in CATEGORICAL if c in X_train.columns]
    numeric = [c for c in NUMERIC if c in X_train.columns]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        estimator = RandomForestClassifier(n_estimators=300, random_state=42,
                                           class_weight="balanced", min_samples_leaf=2)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

st.title("❤️ Heart Disease Risk Classification")
st.caption("Educational machine-learning demonstration using the UCI Heart Disease dataset.")

st.warning("This app is for learning and demonstration only. It is not a medical device, does not diagnose heart disease, and must not be used to make healthcare decisions.")

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Choose algorithm", ["Random Forest", "Logistic Regression"])
    test_size = st.slider("Test set (%)", min_value=15, max_value=40, value=20, step=5)
    st.markdown("---")
    st.markdown("**Dataset:** UCI Heart Disease (Cleveland and related records)")
    st.markdown("[Open official UCI dataset page](https://archive.ics.uci.edu/dataset/45/heart+disease)")

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.info("Check your internet connection for the first run, or download the UCI dataset and save it as data/heart_disease.csv. See README.md.")
    st.stop()

X = df[FEATURES]
y = df["target_binary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, stratify=y
)
model = build_model(model_name, X_train)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

m1, m2, m3 = st.columns(3)
m1.metric("Dataset records", f"{len(df):,}")
m2.metric("Input features", len(FEATURES))
m3.metric("Test accuracy", f"{accuracy:.1%}")

tab1, tab2, tab3 = st.tabs(["Predict a sample", "Model evaluation", "Dataset preview"])

with tab1:
    st.subheader("Enter sample measurements")
    st.write("Values below are example inputs. Use values from a properly documented dataset or clinician-provided measurements for educational testing.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 50)
            sex = st.selectbox("Sex code", [0, 1], format_func=lambda x: "0 — female" if x == 0 else "1 — male")
            cp = st.selectbox("Chest pain type (cp)", [1, 2, 3, 4], help="UCI encoded category; see dataset documentation.")
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 250, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 80, 700, 220)
        with c2:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dL (fbs)", [0, 1])
            restecg = st.selectbox("Resting ECG code (restecg)", [0, 1, 2])
            thalach = st.number_input("Maximum heart rate achieved", 60, 250, 150)
            exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
        with c3:
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
            slope = st.selectbox("Peak exercise ST slope code (slope)", [1, 2, 3])
            ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])
            thal = st.selectbox("Thal code (thal)", [3, 6, 7])
        submitted = st.form_submit_button("Run model prediction", type="primary")
    if submitted:
        sample = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }], columns=FEATURES)
        result = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        positive_index = list(model.classes_).index(1)
        probability = float(probabilities[positive_index])
        st.subheader("Model output")
        st.metric("Estimated model probability for the positive class", f"{probability:.1%}")
        if result == 1:
            st.error("The model classified this record as: heart disease present in the training label.")
        else:
            st.success("The model classified this record as: no heart disease in the training label.")
        st.caption("This is a dataset-based model output, not an individual's clinical risk or diagnosis. A low model score does not rule out disease.")

with tab2:
    st.subheader("Held-out test set")
    st.write(f"Algorithm: **{model_name}** · Split: **{100-test_size}% training / {test_size}% testing**")
    st.text("Classification report")
    report = classification_report(y_test, pred, target_names=["No disease (0)", "Disease present (1)"], zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No disease", "Disease present"]).plot(ax=ax, colorbar=False)
    st.pyplot(fig)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, list(model.classes_).index(1)])
        st.metric("ROC-AUC", f"{auc:.3f}")
    except Exception:
        st.info("ROC-AUC could not be calculated for this split.")

with tab3:
    st.subheader("Dataset preview")
    st.write("The original target is converted to binary: 0 = no disease; 1 = target values 1–4 combined as disease present.")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Class counts")
    st.bar_chart(df["target_binary"].value_counts().sort_index().rename(index={0: "No disease", 1: "Disease present"}))
    st.download_button("Download loaded dataset CSV", df.to_csv(index=False).encode("utf-8"), "heart_disease_binary.csv", "text/csv")

st.markdown("---")
st.caption("Dataset credit: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989), UCI Machine Learning Repository, Heart Disease, DOI: 10.24432/C52P4X.")
