# Dataset

This project uses the **UCI Heart Disease Dataset** for educational and research purposes.

## Source

- **Official UCI Machine Learning Repository**: https://archive.ics.uci.edu/ml/datasets/heart+disease
- **Primary data file**: `data/raw/heart.csv` (Cleveland database)
- **Additional databases**: Hungarian, Swiss, and Long Beach VA datasets also available in the UCI repository

## Files

| File | Description |
|---|---|
| `heart.csv` | Cleveland database, 303 instances, 14 attributes |
| `heart-disease.names` | Attribute names and descriptions |

## Attribute Information

The dataset contains 14 attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num), where:

- `num` is the diagnosis target (0 = no disease, 1-4 = presence of disease)
- Target is transformed to binary: 0 vs. (1, 2, 3, 4) for model training

See `heart-disease.names` for full attribute documentation.

## Preprocessing

The `train_model.py` script performs the following:

1. Loads the UCI Heart Disease dataset via `ucimlrepo`
2. Converts the target from multi-class (0-4) to binary classification (0 = no disease, 1 = disease present)
3. Selects 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
4. Splits data 80/20 train/test with `stratify=y` and `random_state=42`
5. Fits a pipeline with `SimpleImputer(strategy="median")` + `StandardScaler()` + `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)`
6. Saves the fitted pipeline to `models/heart_model.joblib`

## Licensing

This dataset is hosted on the UCI Machine Learning Repository and is intended for research and educational use. See the UCI website for full licensing information.

## Usage

Do not use this dataset or model for clinical diagnosis, treatment decisions, or emergency assessment. This project is an educational prototype only.