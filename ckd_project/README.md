# A Hybrid Machine Learning and Deep Learning Approach for Chronic Kidney Disease Prediction

This project predicts Chronic Kidney Disease (CKD) using a **stacking ensemble**
that combines two classical Machine Learning models with one Deep Learning
model, then serves predictions through a **Flask** web application.

## Architecture

```
                ┌───────────────────┐
                │   Patient Inputs    │
                │ (24 clinical/lab    │
                │     features)        │
                └─────────┬──────────┘
                          │
            ┌─────────────┼──────────────┐
            ▼              ▼               ▼
     ┌───────────┐  ┌───────────┐  ┌────────────────┐
     │  Random    │  │  XGBoost   │  │  Deep Neural    │
     │  Forest    │  │  (ML)      │  │  Network (DL)   │
     │  (ML)      │  │            │  │  Keras/TF ANN   │
     └─────┬──────┘  └─────┬──────┘  └────────┬────────┘
           │  prob          │  prob            │  prob
           └────────────────┼──────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   Meta-Learner        │
                  │ (Logistic Regression) │
                  └──────────┬───────────┘
                             ▼
                    Final CKD / No-CKD
                        prediction
```

Each base learner is trained with 5-fold cross-validation to produce
out-of-fold probability predictions ("meta-features"). The meta-learner
(Logistic Regression) is trained on those meta-features, learning how much
to trust each base model. This is the "hybrid" part of the project: ML and
DL models are combined rather than used in isolation.

## Project structure

```
ckd_project/
├── data/
│   └── chronic_kidney_disease.csv   <- put the dataset here (see below)
├── model/                            <- created automatically by train_model.py
│   ├── rf_model.joblib
│   ├── xgb_model.joblib
│   ├── dl_model.keras
│   ├── meta_model.joblib
│   ├── scaler.joblib
│   ├── label_encoders.joblib
│   ├── cat_imputer.joblib
│   ├── num_imputer.joblib
│   └── metadata.json
├── templates/
│   ├── index.html                    <- input form
│   └── result.html                   <- prediction result page
├── train_model.py                    <- trains and saves the hybrid model
├── app.py                            <- Flask web app (inference)
├── requirements.txt
└── README.md
```

## 1. Get the dataset

This project is built for the standard **UCI Chronic Kidney Disease dataset**
(400 rows, 24 features + class label). Download it from the UCI Machine
Learning Repository ("Chronic_Kidney_Disease" dataset) and save it as:

```
data/chronic_kidney_disease.csv
```

The file should have **no header row** — `train_model.py` assigns the
correct column names automatically. If your downloaded copy already has a
header row, just delete that first line before saving.

If you have your own CKD dataset with different column names, you'll need to
update the `COLUMN_NAMES`, `NUMERIC_COLS`, and `CATEGORICAL_COLS` lists near
the top of `train_model.py` to match.

## 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the hybrid model

```bash
python train_model.py
```

This will:
- load and clean the dataset
- impute missing values and encode/scale features
- train Random Forest, XGBoost, and the ANN with 5-fold cross-validation to
  build the stacking meta-features
- train the Logistic Regression meta-learner on top
- print accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix for
  both the final hybrid model and each individual base learner (useful for
  your project report, to show the hybrid model's improvement)
- save every trained artifact into `model/`

Typical training time is under a minute on a laptop CPU, since the dataset
is small (~400 rows).

## 4. Run the Flask app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Fill in the form with
patient values and click "Run Assessment" to see the prediction along with
a breakdown of what each individual model (RF, XGBoost, DL) predicted, and
the final combined hybrid score.

## Notes for your report / viva

- The dataset has many missing values (a known property of the UCI CKD
  dataset); these are handled with median imputation (numeric) and
  most-frequent imputation (categorical) — implemented with `SimpleImputer`.
- Numeric features are standardized with `StandardScaler` before being fed
  to the DL model, since neural networks are sensitive to feature scale.
- Stacking uses out-of-fold predictions (via `StratifiedKFold`) specifically
  to avoid data leakage into the meta-learner — a point worth mentioning if
  asked about your methodology.
- `train_model.py` prints standalone metrics for each base model as well as
  the combined hybrid model, so you can directly cite the accuracy
  improvement of the hybrid approach over any single model in your report.
- The Flask app's dropdown options are read directly from the trained
  encoders (not hardcoded), so the form can never submit a category value
  the model wasn't trained on.

## Disclaimer

This is an academic/educational project. Predictions are generated by a
statistical model and are not a substitute for professional medical
diagnosis.
