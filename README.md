# Student Math Score Prediction — ML + Flask

End-to-end machine learning project that **predicts a student’s `math_score`** from demographic and prior test features. It includes a **training pipeline** (ingestion → preprocessing → model selection) and a **Flask web app** for interactive predictions.

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Architecture & data flow](#architecture--data-flow)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the model](#training-the-model)
- [Running the web app](#running-the-web-app)
- [API / routes](#api--routes)
- [Model & preprocessing details](#model--preprocessing-details)
- [Troubleshooting](#troubleshooting)
- [License & credits](#license--credits)

---

## Features

- **Regression** target: **`math_score`**
- **Input features** (from form / CSV): `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`
- **Preprocessing**: sklearn `ColumnTransformer` with numeric + categorical pipelines (imputation, scaling, one-hot encoding)
- **Model selection**: multiple regressors with `GridSearchCV`; best model chosen by **test R²** (see `src/utils.py`)
- **Persistence**: `dill`-serialized `preprocessor.pkl` and `model.pkl` under `artifacts/`
- **Web UI**: Flask + Jinja2 templates (`templates/`)

---

## Tech stack

| Area | Libraries |
|------|-----------|
| Web | Flask |
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, CatBoost |
| Serialization | dill |
| Viz (notebook) | matplotlib, seaborn |

See `requirements.txt` for the full list.

---

## Project structure

```text
MLprojct1/
├── app.py                          # Flask entrypoint (routes /, /predict)
├── requirements.txt
├── setup.py                        # Optional: pip install -e .
├── README.md
│
├── src/
│   ├── logger.py                   # Logging configuration
│   ├── exception.py                # CustomException wrapper
│   ├── utils.py                    # save_object, load_object, evaluate_models
│   ├── components/
│   │   ├── data_ingestion.py       # Load CSV, split, kick off training (run as script)
│   │   ├── data_transformation.py  # Fit preprocessor, save preprocessor.pkl
│   │   └── model_trainer.py        # Grid search + best model, save model.pkl
│   └── pipeline/
│       └── predict_pipeline.py     # PredictPipeline + CustomData (inference)
│
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form + results
├── static/                         # CSS for templates
│
├── notebook/
│   └── data/stud.csv             # Source dataset for training pipeline
├── artifacts/                    # Generated: splits, preprocessor.pkl, model.pkl
└── logs/                         # Created at runtime (logging)
```

---

## Architecture & data flow

### 1. Training (offline)

Run `data_ingestion.py` as a script:

```text
notebook/data/stud.csv
    → DataIngestion.initiate_data_ingestion()
        → artifacts/data.csv, train.csv, test.csv
    → DataTransformation.initiate_data_transformation()
        → fit preprocessor on train inputs → artifacts/preprocessor.pkl
    → ModelTrainer.initiate_model_trainer()
        → GridSearchCV per model → pick best test R² → artifacts/model.pkl
```

**Entry point:** `src/components/data_ingestion.py` (`if __name__ == "__main__":`).

### 2. Inference (web)

```text
Browser POST /predict (form fields)
    → CustomData.get_data_as_dataframe()   # single-row DataFrame
    → PredictPipeline.predict()
        → load preprocessor.pkl + model.pkl
        → preprocessor.transform(features)
        → model.predict(...)
    → home.html shows predicted math_score
```

**Entry point:** `app.py`.

### Call graph (summary)

| Caller | Callee | Role |
|--------|--------|------|
| `app.py` | `CustomData`, `PredictPipeline` | Build input + run inference |
| `PredictPipeline` | `load_object` | Load pickles from `artifacts/` |
| `data_ingestion` (`__main__`) | `DataTransformation`, `ModelTrainer` | Full training run |
| `DataTransformation` | `save_object` | Save `preprocessor.pkl` |
| `ModelTrainer` | `evaluate_models`, `save_object` | Train + save `model.pkl` |

---

## Prerequisites

- **Python 3.10+** recommended (adjust if your environment differs)
- Dataset file: **`notebook/data/stud.csv`** must exist before training (columns must include the target `math_score` and all feature columns used in code)

---

## Installation

From the project root (`MLprojct1/`):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

**Optional — install as a package** (helps some IDEs resolve `src` imports):

```bash
pip install -e .
```

---

## Training the model

1. Ensure `notebook/data/stud.csv` is present.
2. From project root:

```bash
python src/components/data_ingestion.py
```

This writes:

- `artifacts/data.csv`, `artifacts/train.csv`, `artifacts/test.csv`
- `artifacts/preprocessor.pkl`
- `artifacts/model.pkl`

**Note:** `ModelTrainer` raises an error if no model achieves **test R² ≥ 0.6** (`model_trainer.py`). If that happens, relax hyperparameters, add data, or lower the threshold in code.

---

## Running the web app

From project root:

```bash
python app.py
```

Default: **http://127.0.0.1:5000** (Flask `debug=True`).

- **`/`** — landing page (`index.html`)
- **`/predict`** — GET shows form (`home.html`); POST returns prediction

Ensure **`artifacts/preprocessor.pkl`** and **`artifacts/model.pkl`** exist (run training first).

---

## API / routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves `index.html` |
| GET | `/predict` | Shows prediction form (`home.html`) |
| POST | `/predict` | Accepts form fields, returns rendered `home.html` with `results` (predicted score) |

Form field names expected by `app.py` / `CustomData`:

- `gender`, `ethnicity` → maps to `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, `reading_score`, `writing_score`

---

## Model & preprocessing details

- **Target column:** `math_score` (only used during training; not passed at inference).
- **Numeric features:** `reading_score`, `writing_score` — median imputation + `StandardScaler`.
- **Categorical features:** `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course` — mode imputation + `OneHotEncoder(handle_unknown='ignore')` + sparse-friendly scaling.
- **Models compared:** Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoost, AdaBoost, KNeighbors — see `src/components/model_trainer.py`.
- **Selection metric:** **Test R²** from `evaluate_models` (best score wins).

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `ModuleNotFoundError: src` | Run commands from project root, or `pip install -e .` |
| Missing `preprocessor.pkl` / `model.pkl` | Run `python src/components/data_ingestion.py` first |
| `No best model found` | Training R² threshold 0.6 not met — tune models/data or threshold |
| Wrong predictions / errors on form submit | Ensure CSV column names match `CustomData` and `data_transformation` |
| `stud.csv` not found | Place dataset at `notebook/data/stud.csv` |

---

## License & credits

- Package metadata: see `setup.py` (`mlproject` v0.0.1).
- This README describes the repository layout and typical workflows; adjust author/license as needed for your use case.

---

## Quick reference

```bash
# Train
python src/components/data_ingestion.py

# Serve
python app.py
```
