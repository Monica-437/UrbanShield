"""
UrbanShield - models.py
-----------------------
Machine-learning models used by UrbanShield.

IMPORTANT:
This module preserves the ML approach used in the previous UrbanTrace
project while making the models reusable for the new UrbanShield pipeline.

Existing ML/analysis components preserved:
- Gaussian Naive Bayes
- Decision Tree
- KNN
- K-Means
- DBSCAN
- Agglomerative Clustering
- Linear Regression

The models are used for:
- Risk-related analysis
- Crime similarity
- Crime clustering
- Hotspot identification
- Trend estimation
- Supporting incoming-incident analysis

The frontend is NOT handled here.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
)


# ============================================================
# GLOBAL MODEL STORAGE
# ============================================================

_models: Dict[str, Any] = {}

_encoders: Dict[str, LabelEncoder] = {}

_scaler: Optional[StandardScaler] = None


# ============================================================
# COMMON FEATURE PREPARATION
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    categorical_columns: Optional[list[str]] = None,
):
    """
    Prepare features for machine-learning models.

    Categorical columns are label encoded.
    Numerical columns are converted to numeric values.
    """

    data = df.copy()

    categorical_columns = categorical_columns or []

    for column in categorical_columns:

        if column not in data.columns:
            continue

        encoder = LabelEncoder()

        values = data[column].fillna("Unknown").astype(str)

        data[column] = encoder.fit_transform(values)

        _encoders[column] = encoder

    for column in features:

        if column not in data.columns:
            data[column] = 0

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        ).fillna(0)

    return data[features]


# ============================================================
# SCALING
# ============================================================

def scale_features(X, fit: bool = True):
    """
    Standardize numerical features.

    Used by distance-based algorithms such as:
    - KNN
    - K-Means
    - DBSCAN
    - Agglomerative Clustering
    """

    global _scaler

    if fit or _scaler is None:

        _scaler = StandardScaler()

        return _scaler.fit_transform(X)

    return _scaler.transform(X)


# ============================================================
# GAUSSIAN NAIVE BAYES
# ============================================================

def train_naive_bayes(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Train Gaussian Naive Bayes.

    Used for classification-based analysis.
    """

    model = GaussianNB()

    model.fit(X, y)

    _models["naive_bayes"] = model

    return model


def predict_naive_bayes(X):
    """
    Predict using the trained Gaussian Naive Bayes model.
    """

    model = _models.get("naive_bayes")

    if model is None:
        raise RuntimeError(
            "Gaussian Naive Bayes model has not been trained."
        )

    prediction = model.predict(X)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)

    return prediction, probability


# ============================================================
# DECISION TREE
# ============================================================

def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: Optional[int] = None,
):
    """
    Train Decision Tree classifier.
    """

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
    )

    model.fit(X, y)

    _models["decision_tree"] = model

    return model


def predict_decision_tree(X):
    """
    Predict using Decision Tree.
    """

    model = _models.get("decision_tree")

    if model is None:
        raise RuntimeError(
            "Decision Tree model has not been trained."
        )

    prediction = model.predict(X)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)

    return prediction, probability


# ============================================================
# K-NEAREST NEIGHBOURS
# ============================================================

def train_knn(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5,
):
    """
    Train KNN classifier.

    KNN is useful for identifying incidents
    similar to previous incidents.
    """

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    model.fit(X, y)

    _models["knn"] = model

    return model


def predict_knn(X):
    """
    Predict using KNN.
    """

    model = _models.get("knn")

    if model is None:
        raise RuntimeError(
            "KNN model has not been trained."
        )

    prediction = model.predict(X)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)

    return prediction, probability


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def run_kmeans(
    X,
    n_clusters: int = 4,
    random_state: int = 42,
):
    """
    Perform K-Means clustering.

    Used to identify groups of similar crime locations
    or incident patterns.
    """

    X_scaled = scale_features(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    _models["kmeans"] = model

    return labels, model


# ============================================================
# DBSCAN HOTSPOT DETECTION
# ============================================================

def run_dbscan(
    X,
    eps: float = 0.5,
    min_samples: int = 5,
):
    """
    Perform DBSCAN clustering.

    DBSCAN is particularly useful for detecting
    dense groups of incidents and potential hotspots.
    """

    X_scaled = scale_features(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(X_scaled)

    _models["dbscan"] = model

    return labels, model


# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================

def run_agglomerative(
    X,
    n_clusters: int = 4,
):
    """
    Perform hierarchical/agglomerative clustering.
    """

    X_scaled = scale_features(X)

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X_scaled)

    _models["agglomerative"] = model

    return labels, model


# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Train Linear Regression.

    Used for numerical trend estimation.
    """

    model = LinearRegression()

    model.fit(X, y)

    _models["linear_regression"] = model

    return model


def predict_linear_regression(X):
    """
    Predict numerical values using Linear Regression.
    """

    model = _models.get("linear_regression")

    if model is None:
        raise RuntimeError(
            "Linear Regression model has not been trained."
        )

    return model.predict(X)


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a classification model using accuracy.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "accuracy": float(accuracy),
    }


def evaluate_regression(model, X_test, y_test):
    """
    Evaluate a regression model using mean squared error.
    """

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


# ============================================================
# MODEL REGISTRY
# ============================================================

def get_model(name: str):
    """
    Retrieve a trained model by name.
    """

    return _models.get(name)


def get_available_models():
    """
    Return the names of currently trained models.
    """

    return list(_models.keys())


# ============================================================
# RESET
# ============================================================

def reset_models():
    """
    Clear trained models and preprocessing objects.
    """

    global _models
    global _encoders
    global _scaler

    _models = {}
    _encoders = {}
    _scaler = None