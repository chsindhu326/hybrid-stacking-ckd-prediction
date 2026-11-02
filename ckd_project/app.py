"""
app.py
=======
Flask web application that serves the hybrid ML + DL CKD prediction model.

Loads all the artifacts saved by train_model.py (RF, XGBoost, DL/ANN,
meta-learner, scaler, imputers, label encoders) and exposes:
  GET  /            -> HTML form for entering patient data
  POST /predict      -> runs the hybrid pipeline and shows the result

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class CompatibleGlorotUniform(keras.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        super().__init__(seed=seed, **kwargs)


def load_dl_model(path: str):
    try:
        return keras.models.load_model(
            path,
            custom_objects={"GlorotUniform": CompatibleGlorotUniform},
        )
    except Exception:
        import train_model
        train_model.main()
        return keras.models.load_model(
            path,
            custom_objects={"GlorotUniform": CompatibleGlorotUniform},
        )
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "chronic_kidney_disease.csv")

app = Flask(__name__)


def ensure_model_artifacts():
    """Train the model if the required artifacts are missing."""
    required_files = [
        os.path.join(MODEL_DIR, "rf_model.joblib"),
        os.path.join(MODEL_DIR, "xgb_model.joblib"),
        os.path.join(MODEL_DIR, "meta_model.joblib"),
        os.path.join(MODEL_DIR, "dl_model.keras"),
        os.path.join(MODEL_DIR, "scaler.joblib"),
        os.path.join(MODEL_DIR, "label_encoders.joblib"),
        os.path.join(MODEL_DIR, "cat_imputer.joblib"),
        os.path.join(MODEL_DIR, "num_imputer.joblib"),
        os.path.join(MODEL_DIR, "metadata.json"),
    ]

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        fallback_path = os.path.join(BASE_DIR, "..", "chronic_kidney_disease.csv")
        if os.path.exists(fallback_path):
            import shutil
            shutil.copy2(fallback_path, DATA_PATH)

    if any(not os.path.exists(path) for path in required_files):
        import train_model
        train_model.main()


ensure_model_artifacts()

# ----------------------------------------------------------------------
# LOAD ALL ARTIFACTS ONCE AT STARTUP
# ----------------------------------------------------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_model.joblib"))
dl_model = load_dl_model(os.path.join(MODEL_DIR, "dl_model.keras"))

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
cat_imputer = joblib.load(os.path.join(MODEL_DIR, "cat_imputer.joblib"))
num_imputer = joblib.load(os.path.join(MODEL_DIR, "num_imputer.joblib"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
FEATURE_COLS = metadata["feature_cols"]
# Use the current metadata threshold; if metadata is missing or stale, default
# to 0.63 so the hybrid model follows the best-performing threshold from
# current evaluation.
DECISION_THRESHOLD = metadata.get("decision_threshold", 0.63)

# Human-friendly labels for the form (used in templates/index.html too)
FIELD_LABELS = {
    "age": "Age (years)",
    "bp": "Blood Pressure (mm/Hg)",
    "sg": "Specific Gravity",
    "al": "Albumin (0-5)",
    "su": "Sugar (0-5)",
    "rbc": "Red Blood Cells",
    "pc": "Pus Cell",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "bgr": "Blood Glucose Random (mg/dl)",
    "bu": "Blood Urea (mg/dl)",
    "sc": "Serum Creatinine (mg/dl)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (gms)",
    "pcv": "Packed Cell Volume",
    "wc": "White Blood Cell Count (cells/cumm)",
    "rc": "Red Blood Cell Count (millions/cumm)",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Pedal Edema",
    "ane": "Anemia",
}

# Dropdown options for categorical fields. These are derived directly from
# each column's fitted LabelEncoder (label_encoders[col].classes_), which
# guarantees the dropdown values shown in the form are EXACTLY the values
# the encoder was trained on -- so a value picked from the dropdown can
# never raise an "unseen label" error at prediction time.
CATEGORICAL_OPTIONS = {
    col: list(label_encoders[col].classes_) for col in CATEGORICAL_COLS
}


def preprocess_input(form_data: dict) -> np.ndarray:
    """
    Converts the raw form input (strings) into a properly scaled/encoded
    feature vector in the exact same order/format the models were trained on.
    Uses pandas DataFrames with the original column names so the imputers
    and scaler (which were fit on DataFrames in train_model.py) behave
    identically at inference time.
    """
    # --- numeric fields ---
    numeric_row = {}
    for col in NUMERIC_COLS:
        val = form_data.get(col, "").strip()
        numeric_row[col] = float(val) if val != "" else np.nan
    numeric_df = pd.DataFrame([numeric_row], columns=NUMERIC_COLS)
    numeric_df[NUMERIC_COLS] = num_imputer.transform(numeric_df[NUMERIC_COLS])
    numeric_df[NUMERIC_COLS] = scaler.transform(numeric_df[NUMERIC_COLS])

    # --- categorical fields ---
    categorical_row = {}
    for col in CATEGORICAL_COLS:
        val = form_data.get(col, "").strip().lower()
        categorical_row[col] = val if val != "" else np.nan
    categorical_df = pd.DataFrame([categorical_row], columns=CATEGORICAL_COLS)
    categorical_df[CATEGORICAL_COLS] = cat_imputer.transform(categorical_df[CATEGORICAL_COLS])

    encoded_cats = []
    for i, col in enumerate(CATEGORICAL_COLS):
        le = label_encoders[col]
        raw_val = categorical_df[col].iloc[0]
        if raw_val in le.classes_:
            encoded_cats.append(le.transform([raw_val])[0])
        else:
            # Defensive fallback: if a value somehow isn't one the encoder
            # saw during training (shouldn't happen via the dropdown form,
            # but could via a direct API call), fall back to the imputer's
            # most-frequent category for that column instead of crashing.
            fallback_val = cat_imputer.statistics_[i]
            encoded_cats.append(le.transform([fallback_val])[0])
    encoded_cats = np.array(encoded_cats).reshape(1, -1)

    # Final feature vector: numeric cols first, then categorical (must match
    # the FEATURE_COLS order used in train_model.py: NUMERIC_COLS + CATEGORICAL_COLS)
    X = np.concatenate([numeric_df[NUMERIC_COLS].values, encoded_cats], axis=1).astype(np.float32)
    return X


def hybrid_predict(X: np.ndarray):
    """
    Runs the full hybrid stacking pipeline on a single feature vector:
    RF + XGBoost + DL base learners -> meta-learner -> final probability.
    """
    rf_proba = rf_model.predict_proba(X)[:, 1]
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    dl_proba = dl_model.predict(X, verbose=0).ravel()

    rf_pred = (rf_proba >= 0.5).astype(int)
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    dl_pred = (dl_proba >= 0.5).astype(int)

    meta_X = np.column_stack([rf_proba, xgb_proba, dl_proba])
    final_proba = meta_model.predict_proba(meta_X)[:, 1][0]
    final_pred = int(final_proba >= DECISION_THRESHOLD)
    final_confidence = float(final_proba if final_pred == 1 else 1.0 - final_proba)

    breakdown = {
        "random_forest": round(float(rf_proba[0] if rf_pred[0] == 1 else 1.0 - rf_proba[0]) * 100, 2),
        "xgboost": round(float(xgb_proba[0] if xgb_pred[0] == 1 else 1.0 - xgb_proba[0]) * 100, 2),
        "deep_learning": round(float(dl_proba[0] if dl_pred[0] == 1 else 1.0 - dl_proba[0]) * 100, 2),
        "hybrid_final": round(final_confidence * 100, 2),
    }
    return final_pred, breakdown, final_proba


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        numeric_cols=NUMERIC_COLS,
        categorical_cols=CATEGORICAL_COLS,
        field_labels=FIELD_LABELS,
        categorical_options=CATEGORICAL_OPTIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = preprocess_input(request.form)
        prediction, breakdown, final_proba = hybrid_predict(X)

        result_text = "CKD Detected" if prediction == 1 else "No CKD Detected"
        result_class = "result-positive" if prediction == 1 else "result-negative"

        return render_template(
            "result.html",
            result_text=result_text,
            result_class=result_class,
            breakdown=breakdown,
            final_proba=final_proba,
        )
    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
