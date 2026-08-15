"""
UrbanShield - anomaly_detection.py
-----------------------------------

Detects unusual crime incidents using Isolation Forest.

Purpose:
    Identify incoming incidents whose combination of characteristics
    is statistically unusual compared with historical crime patterns.

The module provides:
    1. Model training using historical data
    2. Encoding of categorical features
    3. Anomaly scoring for new incidents
    4. Human-readable explanations

The anomaly result is used by pipeline.py and risk_engine.py
to contribute to the dynamic risk assessment.

Important:
    Anomaly detection is a decision-support signal only.
    It does not automatically determine guilt, intent, or required
    law-enforcement action.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

try:
    import streamlit as st
except ImportError:
    st = None


# -------------------------------------------------------------------
# Features used for anomaly detection
# -------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "crime_type",
    "crime_category",
    "crime_severity",
    "district",
    "weapon_used",
    "domestic_related",
    "gang_related",
    "property_damage",
]

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "estimated_loss",
]


# -------------------------------------------------------------------
# Optional Streamlit caching
# -------------------------------------------------------------------

def _cache_resource(func):
    """
    Apply Streamlit resource caching when Streamlit is available.

    This keeps the module usable from both:
        - Streamlit
        - FastAPI
        - command-line/testing environments
    """
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)
    return func


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _clean_value(value):
    """
    Convert missing values to a consistent string representation.
    """
    if value is None:
        return "Unknown"

    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if not value:
        return "Unknown"

    return value


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare historical records for anomaly detection.

    Missing columns are safely created so that the model remains
    compatible with slightly different historical datasets.
    """

    data = df.copy()

    for column in CATEGORICAL_FEATURES:
        if column not in data.columns:
            data[column] = "Unknown"

        data[column] = data[column].apply(_clean_value)

    for column in NUMERIC_FEATURES:
        if column not in data.columns:
            data[column] = 0.0

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        ).fillna(0.0)

    return data


# -------------------------------------------------------------------
# Encoder creation
# -------------------------------------------------------------------

def _fit_encoders(df: pd.DataFrame):
    """
    Train one LabelEncoder for each categorical feature.

    Returns:
        dictionary containing encoders.
    """

    encoders = {}

    for column in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()

        values = df[column].apply(_clean_value).astype(str)

        # Add Unknown so that unseen/missing values can be handled.
        unique_values = sorted(set(values.tolist()) | {"Unknown"})

        encoder.fit(unique_values)

        encoders[column] = encoder

    return encoders


def _safe_encode(value, encoder):
    """
    Encode a categorical value.

    If a new incoming incident contains a category that did not exist
    in the historical training data, it is mapped to Unknown instead
    of causing the API to fail.
    """

    value = _clean_value(value)

    if value in encoder.classes_:
        return int(encoder.transform([value])[0])

    if "Unknown" in encoder.classes_:
        return int(encoder.transform(["Unknown"])[0])

    # Defensive fallback.
    return 0


def _transform_dataframe(
    df: pd.DataFrame,
    encoders: dict
) -> np.ndarray:
    """
    Convert a dataframe into the numerical feature matrix expected
    by Isolation Forest.
    """

    data = _prepare_dataframe(df)

    features = []

    # Categorical features
    for column in CATEGORICAL_FEATURES:
        encoder = encoders[column]

        encoded = data[column].apply(
            lambda value: _safe_encode(value, encoder)
        )

        features.append(encoded.to_numpy(dtype=float))

    # Numeric features
    for column in NUMERIC_FEATURES:
        features.append(
            data[column].to_numpy(dtype=float)
        )

    return np.column_stack(features)


def _transform_incident(
    incident: dict,
    encoders: dict
) -> np.ndarray:
    """
    Convert one incoming incident dictionary into a numerical feature
    vector compatible with the trained anomaly model.
    """

    values = []

    # Categorical features
    for column in CATEGORICAL_FEATURES:
        value = incident.get(column, "Unknown")
        encoder = encoders[column]

        values.append(
            float(_safe_encode(value, encoder))
        )

    # Numeric features
    for column in NUMERIC_FEATURES:
        value = incident.get(column, 0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        values.append(value)

    return np.array(values, dtype=float).reshape(1, -1)


# -------------------------------------------------------------------
# Model training
# -------------------------------------------------------------------

@_cache_resource
def train_anomaly_model(
    historical_df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
):
    """
    Train an Isolation Forest using historical crime records.

    Parameters
    ----------
    historical_df:
        Historical crime dataset.

    contamination:
        Expected proportion of unusual observations.

        Default = 5%.

    random_state:
        Ensures reproducible model behaviour.

    Returns
    -------
    model:
        Trained IsolationForest model.

    encoders:
        Dictionary of categorical LabelEncoders.
    """

    if historical_df is None or historical_df.empty:
        raise ValueError(
            "Historical dataset is empty. "
            "Anomaly detection cannot be trained."
        )

    data = _prepare_dataframe(historical_df)

    encoders = _fit_encoders(data)

    X = _transform_dataframe(data, encoders)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X)

    return model, encoders


# -------------------------------------------------------------------
# Explanation generation
# -------------------------------------------------------------------

def _build_explanation(
    incident: dict,
    anomaly_score: float,
    is_anomaly: bool,
) -> str:
    """
    Create a simple human-readable explanation for the dashboard/API.
    """

    if is_anomaly:
        parts = [
            "The incident profile differs noticeably from common "
            "historical patterns."
        ]

        severity = incident.get("crime_severity")

        if severity:
            parts.append(
                f"Severity is recorded as {severity}."
            )

        weapon = incident.get("weapon_used")

        if weapon not in (None, "", "Unknown", "None"):
            parts.append(
                f"Weapon involvement ({weapon}) contributes to "
                "the unusual profile."
            )

        district = incident.get("district")

        if district:
            parts.append(
                f"The incident occurred in district {district}."
            )

        parts.append(
            f"Anomaly score: {anomaly_score:.3f}."
        )

        return " ".join(parts)

    return (
        "The incident profile is broadly consistent with "
        "historical patterns. "
        f"Anomaly score: {anomaly_score:.3f}."
    )


# -------------------------------------------------------------------
# Score incoming incident
# -------------------------------------------------------------------

def score_incident(
    model,
    encoders: dict,
    incident: dict,
):
    """
    Score one incoming incident using the trained Isolation Forest.

    Returns
    -------
    is_anomaly:
        True when the incident is classified as unusual.

    anomaly_score:
        Continuous Isolation Forest decision score.

        Higher values generally indicate observations that are more
        consistent with the training distribution.

    explanation:
        Human-readable explanation for the dashboard/API.
    """

    if model is None:
        raise ValueError(
            "Anomaly model is not available."
        )

    if not encoders:
        raise ValueError(
            "Anomaly encoders are not available."
        )

    X = _transform_incident(
        incident,
        encoders
    )

    # IsolationForest:
    #   1  -> normal
    #  -1  -> anomaly
    prediction = int(
        model.predict(X)[0]
    )

    # decision_function:
    # Larger = more normal
    # Smaller = more anomalous
    raw_score = float(
        model.decision_function(X)[0]
    )

    is_anomaly = prediction == -1

    # Convert to a user-friendly anomaly intensity.
    #
    # This is NOT presented as a probability.
    # It is only a normalized anomaly indicator.
    anomaly_score = float(
        np.clip(
            0.5 - raw_score,
            0.0,
            1.0
        )
    )

    explanation = _build_explanation(
        incident,
        anomaly_score,
        is_anomaly,
    )

    return (
        bool(is_anomaly),
        round(anomaly_score, 4),
        explanation,
    )


# -------------------------------------------------------------------
# Detailed anomaly information
# -------------------------------------------------------------------

def score_incident_detailed(
    model,
    encoders: dict,
    incident: dict,
):
    """
    Extended version of score_incident().

    Useful when the frontend needs more information than the
    basic pipeline result.
    """

    is_anomaly, anomaly_score, explanation = score_incident(
        model,
        encoders,
        incident,
    )

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "explanation": explanation,
        "severity": incident.get("crime_severity"),
        "crime_type": incident.get("crime_type"),
        "district": incident.get("district"),
    }


# -------------------------------------------------------------------
# Historical anomaly analysis
# -------------------------------------------------------------------

def detect_historical_anomalies(
    historical_df: pd.DataFrame,
    model,
    encoders: dict,
) -> pd.DataFrame:
    """
    Apply the trained anomaly model to historical incidents.

    This function is useful for dashboard analysis and validation.

    Returns a copy of the dataframe containing:

        anomaly_flag
        anomaly_score
    """

    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    data = historical_df.copy()

    X = _transform_dataframe(
        data,
        encoders
    )

    predictions = model.predict(X)
    raw_scores = model.decision_function(X)

    data["anomaly_flag"] = (
        predictions == -1
    )

    data["anomaly_score"] = np.clip(
        0.5 - raw_scores,
        0.0,
        1.0
    ).round(4)

    return data


# -------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------

def anomaly_summary(
    historical_df: pd.DataFrame,
    model,
    encoders: dict,
) -> dict:
    """
    Generate summary statistics for anomaly detection.
    """

    scored_df = detect_historical_anomalies(
        historical_df,
        model,
        encoders,
    )

    if scored_df.empty:
        return {
            "total_incidents": 0,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
        }

    total = len(scored_df)

    anomaly_count = int(
        scored_df["anomaly_flag"].sum()
    )

    anomaly_rate = (
        anomaly_count / total
        if total
        else 0.0
    )

    return {
        "total_incidents": total,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(
            anomaly_rate,
            4
        ),
    }