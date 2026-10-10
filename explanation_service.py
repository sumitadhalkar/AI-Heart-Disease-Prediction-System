"""Model-faithful, per-prediction explanations for the trained pipeline.

The saved pipeline is SimpleImputer -> StandardScaler -> LogisticRegression.
For a fitted logistic regression, the predicted log-odds (logit) of the
positive class is an exact linear sum:

    logit = intercept + sum_i(coefficient_i * standardized_value_i)
    P(class=1) = sigmoid(logit)

Each term `coefficient_i * standardized_value_i` is therefore an exact
decomposition of *this model's* logit for *this input* -- not an
approximation, and not a general statement about disease causation. This
module computes that decomposition directly from the fitted pipeline, so the
explanation always matches whatever model is actually loaded (Cleveland or
combined) rather than describing a hardcoded example.

If the loaded classifier isn't a linear model with `coef_`/`intercept_`
(e.g. the pipeline is retrained with a different algorithm later), this
module reports that no model-faithful explanation is available rather than
fabricating one.
"""

import math


def supports_explanation(pipeline):
    classifier = pipeline.named_steps.get("classifier")
    return classifier is not None and hasattr(classifier, "coef_")


def explain_prediction(pipeline, row_df, features, top_n=5):
    """Return a model-faithful breakdown of one prediction.

    `row_df` is the same single-row DataFrame (in `features` order) that was
    passed to `pipeline.predict`. Returns None if the classifier isn't a
    linear model this method applies to.
    """
    if not supports_explanation(pipeline):
        return None

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    # Pass the DataFrame (not a bare array) into the imputer: it was fit on
    # named columns and otherwise warns about missing feature names.
    raw_values = row_df[features].to_numpy(dtype=float)
    imputed = imputer.transform(row_df[features])
    standardized = scaler.transform(imputed)[0]

    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    contributions = []
    for i, feature in enumerate(features):
        contribution = float(coefficients[i] * standardized[i])
        contributions.append(
            {
                "feature": feature,
                "raw_value": float(raw_values[0][i]),
                "standardized_value": float(standardized[i]),
                "coefficient": float(coefficients[i]),
                "contribution": contribution,
                "direction": (
                    "increases" if contribution > 1e-9
                    else "decreases" if contribution < -1e-9
                    else "negligible"
                ),
            }
        )

    logit = intercept + sum(c["contribution"] for c in contributions)
    probability_class1 = 1.0 / (1.0 + math.exp(-logit))

    ranked = sorted(contributions, key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "intercept": intercept,
        "logit": logit,
        "probability_class1": probability_class1,
        "contributions": ranked,
        "top_contributions": ranked[:top_n],
    }
