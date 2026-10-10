"""Plain-language explanations and reference-range comparisons for each
input feature, used to turn a raw prediction into a readable breakdown of
what was entered and how it compares to standard adult clinical guidelines.

Scope and honesty note: most of the thresholds below (blood pressure,
cholesterol, fasting blood sugar, ECG, angina, ST depression, vessel count,
thalassemia result) are standard ADULT clinical guideline cutoffs (AHA/ACC
blood pressure categories, NCEP ATP III cholesterol categories, standard
cardiology terminology) and do not vary by age or sex. The one value that IS
age-adjusted here is predicted maximum heart rate, using the standard
"220 - age" estimate from exercise physiology. There is no dataset or
guideline backing an age-specific "normal" value for every one of these 13
features, so this module does not invent one -- it tells the user plainly
which comparisons are age-adjusted and which are general adult reference
ranges.

This is general health education content, not a personalized medical
assessment.
"""

SEX_LABELS = {0: "Female", 1: "Male"}
CP_LABELS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic (no chest pain)",
}
FBS_LABELS = {0: "120 mg/dL or below", 1: "Above 120 mg/dL"}
RESTECG_LABELS = {
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Probable or definite left ventricular hypertrophy",
}
EXANG_LABELS = {0: "No", 1: "Yes"}
SLOPE_LABELS = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_LABELS = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}


def predicted_max_heart_rate(age):
    """Standard age-predicted maximum heart rate (220 - age formula)."""
    return 220 - age


def _entry(description, your_value, reference, status, age_adjusted=False):
    return {
        "description": description,
        "your_value": your_value,
        "reference": reference,
        "status": status,
        "age_adjusted": age_adjusted,
    }


def evaluate_feature(feature, value, age):
    """Return a plain-language explanation + reference comparison for one
    feature. `status` is one of "info" (no normal/abnormal judgment applies),
    "normal", "borderline", or "elevated".
    """
    if feature == "age":
        return _entry(
            "Patient age in years.",
            f"{int(value)} years",
            "Used below to estimate your age-predicted maximum heart rate.",
            "info",
        )

    if feature == "sex":
        return _entry(
            "Biological sex as recorded in the dataset.",
            SEX_LABELS.get(int(value), str(value)),
            "Not evaluated as normal/abnormal.",
            "info",
        )

    if feature == "cp":
        code = int(value)
        status = "normal" if code == 4 else "borderline"
        return _entry(
            "Type of chest pain reported.",
            CP_LABELS.get(code, str(value)),
            "No chest pain (asymptomatic) is the reference/healthy pattern.",
            status,
        )

    if feature == "trestbps":
        if value < 120:
            status, ref = "normal", "Normal: below 120 mmHg"
        elif value < 130:
            status, ref = "borderline", "Elevated: 120-129 mmHg"
        elif value < 140:
            status, ref = "elevated", "High (stage 1): 130-139 mmHg"
        else:
            status, ref = "elevated", "High (stage 2): 140 mmHg or above"
        return _entry(
            "Resting blood pressure, measured in mmHg.",
            f"{value:g} mmHg",
            ref,
            status,
        )

    if feature == "chol":
        if value < 200:
            status, ref = "normal", "Desirable: below 200 mg/dL"
        elif value < 240:
            status, ref = "borderline", "Borderline high: 200-239 mg/dL"
        else:
            status, ref = "elevated", "High: 240 mg/dL or above"
        return _entry(
            "Serum cholesterol level, measured in mg/dL.",
            f"{value:g} mg/dL",
            ref,
            status,
        )

    if feature == "fbs":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Whether fasting blood sugar is above 120 mg/dL.",
            FBS_LABELS.get(code, str(value)),
            "120 mg/dL or below is the reference/healthy range.",
            status,
        )

    if feature == "restecg":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Resting electrocardiogram (ECG) result.",
            RESTECG_LABELS.get(code, str(value)),
            "A normal resting ECG is the reference/healthy result.",
            status,
        )

    if feature == "thalach":
        predicted = predicted_max_heart_rate(age)
        pct = (value / predicted * 100) if predicted > 0 else 0
        if pct >= 90:
            status = "normal"
        elif pct >= 70:
            status = "borderline"
        else:
            status = "elevated"
        return _entry(
            "Maximum heart rate achieved during exercise.",
            f"{value:g} bpm ({pct:.0f}% of predicted max)",
            f"Age-predicted maximum for a {int(age)}-year-old: "
            f"{predicted} bpm (220 - age). Reaching close to or above this "
            f"is generally considered a good exercise response.",
            status,
            age_adjusted=True,
        )

    if feature == "exang":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Angina (chest pain) triggered by exercise.",
            EXANG_LABELS.get(code, str(value)),
            "No exercise-induced angina is the reference/healthy result.",
            status,
        )

    if feature == "oldpeak":
        if value <= 0:
            status, ref = "normal", "Normal: 0 (no ST depression)"
        elif value < 2:
            status, ref = "borderline", "Mild: above 0, below 2"
        else:
            status, ref = "elevated", "Significant: 2 or above"
        return _entry(
            "ST depression induced by exercise relative to rest -- a "
            "marker used to detect reduced blood flow to the heart.",
            f"{value:g}",
            ref,
            status,
        )

    if feature == "slope":
        code = int(value)
        status = {1: "normal", 2: "borderline", 3: "elevated"}.get(code, "info")
        return _entry(
            "Slope of the ST segment during peak exercise.",
            SLOPE_LABELS.get(code, str(value)),
            "An upsloping segment is the reference/healthy pattern; "
            "downsloping is most associated with reduced blood flow.",
            status,
        )

    if feature == "ca":
        code = int(value)
        if code == 0:
            status = "normal"
        elif code == 1:
            status = "borderline"
        else:
            status = "elevated"
        return _entry(
            "Number of major vessels (0-3) shown by fluoroscopy.",
            f"{code}",
            "0 vessels showing blockage is the reference/healthy result.",
            status,
        )

    if feature == "thal":
        code = int(value)
        status = "normal" if code == 3 else "elevated"
        return _entry(
            "Thalassemia stress test result.",
            THAL_LABELS.get(code, str(value)),
            "A normal result is the reference/healthy result.",
            status,
        )

    return _entry("", str(value), "", "info")


def build_comparison(values, features, age):
    """Build the per-feature comparison list for the given submitted values.

    `values` maps feature name -> numeric value (as validated by app.py).
    `features` is the ordered list of feature names to include.
    Returns (rows, summary) where `rows` is a list of per-feature dicts
    (feature, label, **evaluate_feature(...)) and `summary` describes how
    many evaluated measurements fell outside the normal reference range.
    """
    rows = []
    for feature in features:
        result = evaluate_feature(feature, values[feature], age)
        rows.append({"feature": feature, **result})

    judged = [r for r in rows if r["status"] != "info"]
    normal_count = sum(1 for r in judged if r["status"] == "normal")
    borderline_count = sum(1 for r in judged if r["status"] == "borderline")
    elevated_count = sum(1 for r in judged if r["status"] == "elevated")

    summary = {
        "total_judged": len(judged),
        "normal_count": normal_count,
        "borderline_count": borderline_count,
        "elevated_count": elevated_count,
    }

    return rows, summary
