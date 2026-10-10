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


STATUS_COLOR_VAR = {
    "normal": "var(--success)",
    "borderline": "var(--warning)",
    "elevated": "var(--danger)",
}


def _gauge(value, scale_min, scale_max, zone_bounds):
    """Build a horizontal mini-gauge for a continuous numeric feature.

    `zone_bounds` is a list of (upper_value, status) pairs in ascending
    order of `upper_value`, covering [scale_min, scale_max]. Used only for
    the visual track/marker -- never for the actual normal/borderline/
    elevated decision, which is computed exactly in evaluate_feature().
    """
    span = scale_max - scale_min
    marker_pct = max(0.0, min(100.0, (value - scale_min) / span * 100))

    stops = []
    prev_pct = 0.0
    for upper_value, status in zone_bounds:
        upper_pct = max(0.0, min(100.0, (upper_value - scale_min) / span * 100))
        color = STATUS_COLOR_VAR[status]
        stops.append(f"{color} {prev_pct:.1f}%")
        stops.append(f"{color} {upper_pct:.1f}%")
        prev_pct = upper_pct
    track_gradient = f"linear-gradient(to right, {', '.join(stops)})"

    return {
        "scale_min": scale_min,
        "scale_max": scale_max,
        "marker_pct": round(marker_pct, 1),
        "track_gradient": track_gradient,
    }


def _entry(
    description, your_value, reference, status,
    age_adjusted=False, group="profile", gauge=None,
):
    return {
        "description": description,
        "your_value": your_value,
        "reference": reference,
        "status": status,
        "age_adjusted": age_adjusted,
        "group": group,
        "gauge": gauge,
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
            group="profile",
        )

    if feature == "sex":
        return _entry(
            "Biological sex as recorded in the dataset.",
            SEX_LABELS.get(int(value), str(value)),
            "Not evaluated as normal/abnormal.",
            "info",
            group="profile",
        )

    if feature == "cp":
        code = int(value)
        status = "normal" if code == 4 else "borderline"
        return _entry(
            "Type of chest pain reported.",
            CP_LABELS.get(code, str(value)),
            "No chest pain (asymptomatic) is the reference/healthy pattern.",
            status,
            group="symptoms",
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
        gauge = _gauge(value, 80, 200, [
            (120, "normal"), (130, "borderline"), (200, "elevated"),
        ])
        return _entry(
            "Resting blood pressure, measured in mmHg.",
            f"{value:g} mmHg",
            ref,
            status,
            group="vitals",
            gauge=gauge,
        )

    if feature == "chol":
        if value < 200:
            status, ref = "normal", "Desirable: below 200 mg/dL"
        elif value < 240:
            status, ref = "borderline", "Borderline high: 200-239 mg/dL"
        else:
            status, ref = "elevated", "High: 240 mg/dL or above"
        gauge = _gauge(value, 100, 350, [
            (200, "normal"), (240, "borderline"), (350, "elevated"),
        ])
        return _entry(
            "Serum cholesterol level, measured in mg/dL.",
            f"{value:g} mg/dL",
            ref,
            status,
            group="vitals",
            gauge=gauge,
        )

    if feature == "fbs":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Whether fasting blood sugar is above 120 mg/dL.",
            FBS_LABELS.get(code, str(value)),
            "120 mg/dL or below is the reference/healthy range.",
            status,
            group="vitals",
        )

    if feature == "restecg":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Resting electrocardiogram (ECG) result.",
            RESTECG_LABELS.get(code, str(value)),
            "A normal resting ECG is the reference/healthy result.",
            status,
            group="cardiac_test",
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
        gauge = _gauge(pct, 40, 120, [
            (70, "elevated"), (90, "borderline"), (120, "normal"),
        ])
        return _entry(
            "Maximum heart rate achieved during exercise.",
            f"{value:g} bpm ({pct:.0f}% of predicted max)",
            f"Age-predicted maximum for a {int(age)}-year-old: "
            f"{predicted} bpm (220 - age). Reaching close to or above this "
            f"is generally considered a good exercise response.",
            status,
            age_adjusted=True,
            group="cardiac_test",
            gauge=gauge,
        )

    if feature == "exang":
        code = int(value)
        status = "normal" if code == 0 else "elevated"
        return _entry(
            "Angina (chest pain) triggered by exercise.",
            EXANG_LABELS.get(code, str(value)),
            "No exercise-induced angina is the reference/healthy result.",
            status,
            group="symptoms",
        )

    if feature == "oldpeak":
        if value <= 0:
            status, ref = "normal", "Normal: 0 (no ST depression)"
        elif value < 2:
            status, ref = "borderline", "Mild: above 0, below 2"
        else:
            status, ref = "elevated", "Significant: 2 or above"
        gauge = _gauge(value, 0, 6, [
            (0.3, "normal"), (2, "borderline"), (6, "elevated"),
        ])
        return _entry(
            "ST depression induced by exercise relative to rest -- a "
            "marker used to detect reduced blood flow to the heart.",
            f"{value:g}",
            ref,
            status,
            group="cardiac_test",
            gauge=gauge,
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
            group="cardiac_test",
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
            group="cardiac_test",
        )

    if feature == "thal":
        code = int(value)
        status = "normal" if code == 3 else "elevated"
        return _entry(
            "Thalassemia stress test result.",
            THAL_LABELS.get(code, str(value)),
            "A normal result is the reference/healthy result.",
            status,
            group="cardiac_test",
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


# Generic, non-prescriptive talking points for flagged measurements. These
# only ever suggest discussing a result with a doctor -- never a specific
# treatment, dose, or diagnosis.
_TALKING_POINTS = {
    "trestbps": ("Blood Pressure", "Your resting blood pressure reading is above the desirable range. Ask your doctor whether follow-up monitoring or further evaluation is appropriate."),
    "chol": ("Cholesterol", "Your cholesterol reading is above the desirable range. Ask your doctor about lipid management and whether further testing is needed."),
    "fbs": ("Blood Sugar", "Your fasting blood sugar reading is above 120 mg/dL. Ask your doctor about screening for diabetes or prediabetes."),
    "cp": ("Chest Pain", "Describe the chest pain pattern you reported in detail to your doctor, including when it occurs and how long it lasts."),
    "exang": ("Exercise-Induced Angina", "Chest pain during exercise should be discussed with your doctor, especially if it is new or worsening."),
    "restecg": ("Resting ECG", "Your resting ECG showed a finding other than normal. Ask your doctor to review this result with you."),
    "thalach": ("Exercise Capacity", "Your measured maximum heart rate was notably below the age-predicted estimate. Ask your doctor whether this needs follow-up."),
    "oldpeak": ("ST Depression (Oldpeak)", "This exercise-test finding can indicate reduced blood flow during exertion. Ask your doctor to explain what this result means for you."),
    "slope": ("ST Segment Slope", "This exercise-test pattern is one of several findings used to assess blood flow. Discuss it with your doctor alongside your other results."),
    "ca": ("Vessels Shown by Fluoroscopy", "This imaging finding should be reviewed with your doctor or cardiologist as part of your overall evaluation."),
    "thal": ("Thalassemia Test Result", "This result is outside the normal pattern. Ask your doctor or cardiologist to explain what it means in your case."),
}

_PRIORITY_BY_STATUS = {"elevated": 3, "borderline": 2}


def build_talking_points(rows, max_items=6):
    """Build a short, generic "talk to your doctor about" list from the
    comparison rows. Only ever suggests discussing a result with a doctor --
    never a specific treatment, dose, or diagnosis. Always leads with a
    fixed urgent-symptom safety note regardless of what was flagged.
    """
    points = [
        {
            "label": "Urgent Symptoms",
            "priority": 4,
            "note": (
                "If you are currently experiencing chest pain, severe "
                "shortness of breath, or fainting, seek urgent medical care "
                "immediately -- do not wait based on this or any automated "
                "result."
            ),
        }
    ]

    for row in rows:
        mapping = _TALKING_POINTS.get(row["feature"])
        priority = _PRIORITY_BY_STATUS.get(row["status"])
        if mapping is None or priority is None:
            continue
        label, note = mapping
        points.append({"label": label, "priority": priority, "note": note})

    if len(points) == 1:
        points.append(
            {
                "label": "Keep It Up",
                "priority": 1,
                "note": (
                    "None of the measurements compared above fell outside "
                    "the reference range used here. Continue regular "
                    "checkups with your doctor."
                ),
            }
        )

    points.sort(key=lambda p: p["priority"], reverse=True)
    return points[:max_items]
