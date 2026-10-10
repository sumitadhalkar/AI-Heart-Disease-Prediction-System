# AI-Based Heart Disease Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-orange.svg)](https://palletsprojects.com/p/flask/)

## Project Overview

**AI-Based Heart Disease Prediction System** is a machine learning project that analyzes a person's health parameters to estimate their risk of heart disease. It is designed as an educational prototype for a final-year computer science project, not as a clinical diagnostic tool.

The system uses the UCI Heart Disease dataset and a Logistic Regression classification pipeline to predict whether a patient's health parameters patterns correspond to the presence or absence of heart disease in the dataset's labeling.

## Purpose of the Machine Learning Model

The trained model takes 13 health parameters as input and outputs a binary classification:
- **Class 0**: The model assigned the input to the negative class (patterns associated with no disease in training data)
- **Class 1**: The model identified patterns associated with the positive class (dataset-labeled disease presence)

**Important**: The output class is **not** a medical diagnosis, a validated personal risk score, or proof that heart disease is present or absent. A qualified healthcare professional must assess any health concerns using appropriate clinical evaluation.

## Features

- **Flask-based web application** with patient input form and prediction results page
- **Patient health parameter input form** with 13 features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Machine learning prediction pipeline** using Scikit-learn Logistic Regression with median imputation and StandardScaler
- **Input validation** on server side (range checks, type checks, whole-number checks)
- **Model evaluation metrics** reported from held-out test set: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- **Responsive HTML/CSS interface** that works from mobile to desktop
- **Model information section** showing algorithm, dataset source, and actual evaluation metrics
- **Clear educational-use and medical-safety notices** throughout the application

## Technology Stack

| Category | Details |
|---|---|
| **Language** | Python 3.11+ |
| **Web Framework** | Flask >= 3.0, < 4.0 |
| **Data Processing** | Pandas >= 2.1, NumPy >= 1.26 |
| **Machine Learning** | Scikit-learn >= 1.4 |
| **Model Persistence** | Joblib >= 1.3 |
| **Dataset Client** | ucimlrepo >= 0.0.7 |
| **Frontend** | HTML5, CSS3 (custom, no frontend framework) |
| **Dataset** | UCI Heart Disease Repository (Cleveland database) |

## Project Structure

```
heart_disease_ai/
├── app.py | Flask backend with input validation, request handling, and prediction pipeline
├── train_model.py | Downloads UCI dataset, trains Logistic Regression pipeline, saves model
├── requirements.txt | Third-party runtime and training dependencies
├── README.md | Project documentation
├── .gitignore | Git exclusion patterns
├── data/
│   ├── raw/ | Original UCI dataset (heart.csv + attribute names)
│   └── processed/ | (Not currently used; raw data used for training)
├── models/
│   └── heart_model.joblib | Trained pipeline (imputer + scaler + LogisticRegression)
├── templates/
│   ├── index.html | Patient input form with categorized groups
│   └── result.html | Prediction result card with medical limitation notice
└── static/
    ├── style.css | Design tokens, responsive layout, component styles
    └── images/ | (Optional: project icons/illustrations)
```

### Key Files Explained

- **`app.py`**: Flask app with routes for the input form (`GET/POST /`) and results page (`GET /result`). Loads the model pipeline from `models/heart_model.joblib`, validates all 13 input features, and renders prediction results with proper medical disclaimers.

- **`train_model.py`**: Script that downloads the UCI Heart Disease dataset (id=45), converts the multi-class target (0-4) to binary classification, trains a Logistic Regression pipeline with `SimpleImputer(strategy="median")` + `StandardScaler()`, evaluates on a 20% held-out test set, and saves the complete pipeline with joblib. Also saves a copy of the raw dataset to `data/raw/heart.csv`.

- **`train_model.py`**: Uses `Path(__file__).resolve().parent` for all paths, so it works regardless of the current working directory.

- **`models/heart_model.joblib`**: Saved pipeline dictionary containing `pipeline`, `features`, `target_definition`, `test_size`, and `random_state`. The pipeline includes preprocessing (imputer + scaler) and the trained classifier.

- **`templates/index.html`**: Patient input form with 5 categorized groups (Demographics, Chest Pain & Vital Signs, Fasting & ECG, Heart Rate & Angina, ST Segment & Vessels). Uses dropdowns for categorical variables, number inputs for continuous features, and inline validation error messages.

- **`templates/result.html`**: Prediction result card with icon, classification title, explanation text, output class code, and strong medical limitation notice.

- **`static/style.css`**: Custom CSS with design tokens (colors, fonts, radii, shadows), responsive grid layouts, focus-visible states, reduced-motion support, and component styles (buttons, cards, notices, form grid, info panels).

- **`data/README.md`**: Documentation of the dataset source, attribute information, preprocessing steps, and licensing.

## Dataset

- **Source**: UCI Machine Learning Repository, Heart Disease Dataset (Cleveland database)
- **URL**: https://archive.ics.uci.edu/ml/datasets/heart+disease
- **Instances**: 303 (Cleveland database)
- **Attributes**: 14 (13 features + 1 target)
- **Target transformation**: Original target values 0-4 are converted to binary: 0 = no disease, 1 = disease present (values 1-4 combined into positive class)
- **Features used**: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- **Preprocessing**: Median imputation for missing values, StandardScaler for feature scaling, 80/20 stratified train/test split (random_state=42)
- **Licensing**: See UCI website for full terms. This dataset is intended for research and educational use only.

## Installation and Setup

### 1. Install Python

Ensure you have Python 3.11 or 3.12 installed.

### 2. Clone the repository

```bash
git clone https://github.com/your-username/ai-heart-disease-prediction.git
cd ai-heart-disease-prediction/heart_disease_ai
```

### 3. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the machine learning model

```bash
python train_model.py
```

*This step requires internet access to download the UCI Heart Disease dataset. The first run will:*
- *Download the dataset via ucimlrepo*
- *Train the Logistic Regression model*
- *Save the pipeline to `models/heart_model.joblib`*
- *Save raw data to `data/raw/heart.csv`*

### 6. Start the Flask server

```bash
python app.py
```

### 7. Open the application

Open your browser and visit:

```
http://127.0.0.1:5000
```

## Usage

1. **Enter patient health data** in the form using the following feature encodings:

   | Feature | Description | Encoding |
   |---|---|---|
   | `age` | Age in years | 1-120, whole number |
   | `sex` | Sex code | 0 = Female, 1 = Male |
   | `cp` | Chest pain type | 1-4 (1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic) |
   | `trestbps` | Resting blood pressure (mm Hg) | 50-300 |
   | `chol` | Cholesterol (mg/dL) | 50-800 |
   | `fbs` | Fasting blood sugar code | 0 = ≤ 120 mg/dL, 1 = > 120 mg/dL |
   | `restecg` | Resting ECG results | 0-2 (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy) |
   | `thalach` | Maximum heart rate achieved | 30-250 |
   | `exang` | Exercise-induced angina | 0 = No, 1 = Yes |
   | `oldpeak` | ST depression induced by exercise | -5 to 15, decimal allowed |
   | `slope` | Peak exercise ST slope | 1-3 (1=Upsloping, 2=Flat, 3=Downsloping) |
   | `ca` | Major vessels colored by fluoroscopy | 0-3 |
   | `thal` | Thalassemia category | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect |

2. **Click "Run Model Prediction"** to submit the form.

3. **View the result** on the prediction results page, which shows:
   - The model's classification (class 0 or class 1)
   - An explanation of what the classification means
   - A prominent medical limitation notice
   - A link to return to the input form

**Note**: Fill in all fields before submitting. Invalid values will be caught with server-side validation and displayed as error messages.

## Model Training and Evaluation

### Algorithm

The model is a **Logistic Regression** classifier with L2 regularization, trained using the `liblinear` solver (default for binary classification with small datasets). The pipeline includes:

- `SimpleImputer(strategy="median")` - Handles missing feature values
- `StandardScaler()` - Centers and scales features to unit variance
- `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)` - Trained with balanced class weights

### Train/Test Split

- 80% training data, 20% held-out test set
- Stratified split to preserve class distribution
- `random_state=42` for reproducibility

### Reported Evaluation Metrics (from actual model training)

| Metric | Value | Description |
|---|---|---|
| **Accuracy** | 86.9% | Fraction of test records classified correctly |
| **Precision** | 81.2% | Fraction of positive predictions that are actually positive in the test set |
| **Recall** | 92.9% | Fraction of positive test records detected by the model |
| **F1-score** | 86.7% | Harmonic mean of precision and recall |
| **ROC-AUC** | 95.0% | How well model scores rank positive records above negative records |
| **Confusion Matrix** | [[27, 6], [2, 26]] | True negatives, false positives, false negatives, true positives |

### Reproducing the Metrics

To reproduce the evaluation metrics:

```bash
python train_model.py
```

The script will output the same metrics to the console. The model pipeline is saved to `models/heart_model.joblib` and can be loaded by `app.py` for predictions.

**Important**: The test set metrics have substantial uncertainty due to the small test set size (61 records). These metrics do not establish clinical performance and should not be used for medical decision-making.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Activate your virtual environment: `source .venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\activate` (Windows), then `pip install -r requirements.txt` |
| `Model not found: Run python train_model.py first.` | Run the training script first: `python train_model.py` |
| Dataset download error | Check your internet connection and retry. The UCI dataset must be downloaded on first run. |
| `Port 5000 is occupied` | Change the port in `app.py`: `app.run(host="127.0.0.1", port=5001, debug=True)`, then visit `http://127.0.0.1:5001` |
| `Unexpected predictions` | Confirm the form uses the same feature order and category encoding as the training dataset. All 13 features must be provided. |
| `ValueError: Dataset is missing expected columns` | Ensure `data/raw/heart.csv` exists and has the correct column names. The training script saves it automatically. |
| ` sklearn.version incompatibility` | Use the specified version constraints in `requirements.txt`. The model was trained with Scikit-learn >= 1.4. |
| `joblib.load errors` | Ensure the model file `models/heart_model.joblib` exists and was saved by this project's `train_model.py`. |

## Testing and Model Verification

### Commands

#### Training the model

```bash
python train_model.py
```

This downloads the UCI Heart Disease dataset (id=45), trains a Logistic Regression
pipeline with `SimpleImputer(strategy="median")` + `StandardScaler()`, evaluates on
a 20% held-out test set, and saves the complete pipeline to
`models/heart_model.joblib`. The raw dataset is also saved to
`data/raw/heart.csv`. Reported metrics (accuracy, precision, recall, F1-score,
ROC-AUC, confusion matrix, classification report) are printed to the console and
also saved to `reports/model_evaluation.json` and `reports/model_evaluation.txt`.

#### Running the sample prediction

```bash
python predict_sample.py
```

This loads the actual saved model, validates a documented sample input using the
exact expected feature names, order, and encodings, runs the model's actual
prediction pipeline, and prints the input values and predicted class. Example output:

```
================================================================
AI HEART DISEASE PREDICTION SYSTEM
================================================================

Model loaded: SUCCESS
Input validation: SUCCESS

Prediction:
  Output class: 0

Status: Sample inference completed successfully.
Note: This is an illustrative software test,
not a medical diagnosis.


The output class above is only an illustration of formatting.
The displayed output class is not an individual's validated
probability of having heart disease.

================================================================
End of prediction output
================================================================
```

If the model file is missing, the script prints an error message and exits.
No patient data or identifiable records are used in the sample input.

#### Running automated tests

```bash
python -m pytest -v
```

This runs the unit test suite in the `tests/` directory. Tests cover:

- Feature names and ordering
- Limits validation
- Synthetic model training and inference
- Model saving and reloading
- Flask home page response
- Successful and failed form submissions
- Graceful handling when the model file is unavailable

Unit tests use synthetic fixtures and do not require downloading the real dataset.
Actual dataset evaluation is a separate integration task.

#### Starting the web application

```bash
python app.py
```

The Flask development server starts at `http://127.0.0.1:5000`. Visit this URL
in your browser to access the prediction interface. For production deployments,
use a production WSGI server and disable debug mode.

### Evaluation Reports

After running `python train_model.py`, the following files are created in
`reports/`:

- `model_evaluation.json` - Structured JSON with all metrics, feature names,
  target definition, test configuration, and the full classification report.
- `model_evaluation.txt` - Human-readable text summary with the same information.
- `confusion_matrix.png` - (Optional) Generated confusion matrix chart.
- `roc_curve.png` - (Optional) Generated ROC curve chart when valid scores are available.

### Health Check

The application provides a `/health` endpoint that reports whether the
application is running and whether the model is loaded:

```bash
curl http://127.0.0.1:5000/health
```

Response example (model loaded):

```json
{"status": "ok", "model": "loaded"}
```

Response example (model not available):

```json
{"status": "degraded", "model": "not_loaded"}
```

The endpoint returns HTTP 200 if the model is loaded, or HTTP 503 if the model
is unavailable. No patient information, secrets, or internal stack traces are
exposed.