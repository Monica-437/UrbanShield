"""
UrbanShield - utils.py
----------------------
Common utility functions used by the UrbanShield backend.

This file handles:
    - Incident validation
    - Coverage checking
    - Incident ID generation
    - Geographic coverage calculation
    - Safe value conversion
    - Date/time validation

IMPORTANT:
This file contains backend utilities only.
It does NOT modify the existing Streamlit frontend or its images.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Tuple
import uuid


# ---------------------------------------------------------------------------
# Custom validation exception
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when an incoming incident fails validation."""
    pass


# ---------------------------------------------------------------------------
# Incident ID
# ---------------------------------------------------------------------------

def new_incident_id() -> str:
    """
    Generate a unique incident ID.

    Example:
        INC-8A21F4C3
    """

    return f"INC-{uuid.uuid4().hex[:8].upper()}"


def new_assessment_id() -> str:
    """Generate a unique risk assessment ID."""

    return f"RISK-{uuid.uuid4().hex[:8].upper()}"


def new_alert_id() -> str:
    """Generate a unique alert ID."""

    return f"ALT-{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------------------------

def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float safely."""

    try:
        if value is None or value == "":
            return default

        result = float(value)

        if result != result:  # NaN check
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Convert a value to integer safely."""

    try:
        if value is None or value == "":
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Date and time validation
# ---------------------------------------------------------------------------

def validate_date(date_value: str) -> bool:
    """
    Validate date in YYYY-MM-DD format.
    """

    try:
        datetime.strptime(str(date_value), "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def validate_time(time_value: str) -> bool:
    """
    Validate time in HH:MM format.
    """

    try:
        datetime.strptime(str(time_value), "%H:%M")
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Geographic coverage
# ---------------------------------------------------------------------------

def compute_coverage_bounds(df) -> dict:
    """
    Calculate the geographic coverage of the historical dataset.

    The result is used to determine whether an incoming incident lies
    inside the geographic area represented by the historical data.

    Returns:
        {
            "min_lat": ...,
            "max_lat": ...,
            "min_lon": ...,
            "max_lon": ...
        }
    """

    if df is None or df.empty:
        return {
            "min_lat": 0.0,
            "max_lat": 0.0,
            "min_lon": 0.0,
            "max_lon": 0.0,
        }

    required = {"latitude", "longitude"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Historical dataset must contain latitude and longitude columns."
        )

    latitudes = df["latitude"].dropna()
    longitudes = df["longitude"].dropna()

    if latitudes.empty or longitudes.empty:
        return {
            "min_lat": 0.0,
            "max_lat": 0.0,
            "min_lon": 0.0,
            "max_lon": 0.0,
        }

    return {
        "min_lat": float(latitudes.min()),
        "max_lat": float(latitudes.max()),
        "min_lon": float(longitudes.min()),
        "max_lon": float(longitudes.max()),
    }


def is_within_coverage(
    latitude: float,
    longitude: float,
    coverage_bounds: Optional[dict],
) -> bool:
    """
    Check whether a geographic coordinate falls inside the
    historical dataset's coverage area.
    """

    if not coverage_bounds:
        return False

    lat = safe_float(latitude)
    lon = safe_float(longitude)

    return (
        coverage_bounds["min_lat"]
        <= lat
        <= coverage_bounds["max_lat"]
        and
        coverage_bounds["min_lon"]
        <= lon
        <= coverage_bounds["max_lon"]
    )


# ---------------------------------------------------------------------------
# Incident validation
# ---------------------------------------------------------------------------

def validate_incident_payload(
    payload: dict,
    valid_crime_types: Optional[Iterable[str]] = None,
    valid_districts: Optional[Iterable[str]] = None,
) -> Tuple[bool, list]:
    """
    Validate an incoming incident before it enters the processing pipeline.

    Returns:
        (True, [])
        OR
        (False, [list of validation errors])
    """

    errors = []

    if not isinstance(payload, dict):
        return False, ["Incident payload must be a dictionary."]

    # -----------------------------------------------------------------------
    # Required fields
    # -----------------------------------------------------------------------

    required_fields = [
        "crime_type",
        "crime_category",
        "crime_severity",
        "latitude",
        "longitude",
        "occurred_date",
        "occurred_time",
    ]

    for field in required_fields:
        value = payload.get(field)

        if value is None or str(value).strip() == "":
            errors.append(f"{field} is required.")

    # Stop deeper validation when required information is missing.
    if errors:
        return False, errors

    # -----------------------------------------------------------------------
    # Crime type
    # -----------------------------------------------------------------------

    crime_type = str(payload.get("crime_type")).strip()

    if valid_crime_types:
        valid_types = {
            str(value).strip()
            for value in valid_crime_types
        }

        if crime_type not in valid_types:
            errors.append(
                f"Invalid crime type: {crime_type}."
            )

    # -----------------------------------------------------------------------
    # District
    # -----------------------------------------------------------------------

    district = payload.get("district")

    if district is not None and str(district).strip() != "":
        district = str(district).strip()

        if valid_districts:
            valid_area_list = {
                str(value).strip()
                for value in valid_districts
            }

            if district not in valid_area_list:
                errors.append(
                    f"Invalid district: {district}."
                )

    # -----------------------------------------------------------------------
    # Coordinates
    # -----------------------------------------------------------------------

    latitude = safe_float(payload.get("latitude"), default=float("nan"))
    longitude = safe_float(payload.get("longitude"), default=float("nan"))

    if latitude != latitude:
        errors.append("Latitude must be a valid number.")

    if longitude != longitude:
        errors.append("Longitude must be a valid number.")

    if latitude == latitude:
        if not -90 <= latitude <= 90:
            errors.append(
                "Latitude must be between -90 and 90."
            )

    if longitude == longitude:
        if not -180 <= longitude <= 180:
            errors.append(
                "Longitude must be between -180 and 180."
            )

    # -----------------------------------------------------------------------
    # Date
    # -----------------------------------------------------------------------

    occurred_date = str(payload.get("occurred_date"))

    if not validate_date(occurred_date):
        errors.append(
            "occurred_date must use YYYY-MM-DD format."
        )

    # -----------------------------------------------------------------------
    # Time
    # -----------------------------------------------------------------------

    occurred_time = str(payload.get("occurred_time"))

    if not validate_time(occurred_time):
        errors.append(
            "occurred_time must use HH:MM format."
        )

    # -----------------------------------------------------------------------
    # Severity
    # -----------------------------------------------------------------------

    allowed_severities = {
        "Infraction",
        "Misdemeanor",
        "Felony",
    }

    severity = str(
        payload.get("crime_severity", "")
    ).strip()

    if severity not in allowed_severities:
        errors.append(
            "crime_severity must be Infraction, Misdemeanor, or Felony."
        )

    # -----------------------------------------------------------------------
    # Boolean-style fields
    # -----------------------------------------------------------------------

    yes_no_fields = [
        "domestic_related",
        "gang_related",
        "property_damage",
    ]

    for field in yes_no_fields:
        value = payload.get(field)

        if value is not None:
            normalized = str(value).strip().lower()

            if normalized not in {"yes", "no"}:
                errors.append(
                    f"{field} must be Yes or No."
                )

    # -----------------------------------------------------------------------
    # Estimated loss
    # -----------------------------------------------------------------------

    estimated_loss = payload.get("estimated_loss", 0)

    try:
        loss = float(estimated_loss)

        if loss < 0:
            errors.append(
                "estimated_loss cannot be negative."
            )

    except (TypeError, ValueError):
        errors.append(
            "estimated_loss must be a valid number."
        )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Normalize incoming incident
# ---------------------------------------------------------------------------

def normalize_incident_payload(payload: dict) -> dict:
    """
    Normalize an incoming incident into a consistent backend format.

    This is especially useful because incidents may arrive from:
        1. Streamlit form
        2. REST API
        3. Future external systems

    The frontend design is not affected.
    """

    normalized = dict(payload)

    normalized["crime_type"] = str(
        payload.get("crime_type", "")
    ).strip()

    normalized["crime_category"] = str(
        payload.get("crime_category", "")
    ).strip()

    normalized["crime_severity"] = str(
        payload.get("crime_severity", "")
    ).strip()

    normalized["district"] = (
        str(payload["district"]).strip()
        if payload.get("district") is not None
        else None
    )

    normalized["neighborhood"] = (
        str(payload["neighborhood"]).strip()
        if payload.get("neighborhood") is not None
        else None
    )

    normalized["latitude"] = safe_float(
        payload.get("latitude")
    )

    normalized["longitude"] = safe_float(
        payload.get("longitude")
    )

    normalized["estimated_loss"] = safe_int(
        payload.get("estimated_loss", 0)
    )

    normalized["weapon_used"] = (
        payload.get("weapon_used")
        or "Unknown"
    )

    normalized["domestic_related"] = (
        payload.get("domestic_related")
        or "No"
    )

    normalized["gang_related"] = (
        payload.get("gang_related")
        or "No"
    )

    normalized["property_damage"] = (
        payload.get("property_damage")
        or "No"
    )

    normalized["priority_level"] = (
        payload.get("priority_level")
        or "Low"
    )

    return normalized


# ---------------------------------------------------------------------------
# Risk-related helpers
# ---------------------------------------------------------------------------

def clamp_score(score: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """
    Keep a numerical score inside a specified range.
    """

    score = safe_float(score)

    return max(
        minimum,
        min(maximum, score)
    )


def risk_level_from_score(score: float) -> str:
    """
    Convert a 0-100 risk score into the UrbanShield risk level.

    Thresholds:
        < 35      LOW
        35-54.9   MODERATE
        55-74.9   HIGH
        >= 75     CRITICAL

    This mirrors the documented risk-engine thresholds.
    """

    score = clamp_score(score)

    if score < 35:
        return "LOW"

    if score < 55:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# ---------------------------------------------------------------------------
# Time-of-day helper
# ---------------------------------------------------------------------------

def get_time_period(time_value: str) -> str:
    """
    Convert HH:MM into a simple time-of-day period.

    Used by trend analysis and decision-support explanations.
    """

    try:
        hour = datetime.strptime(
            str(time_value),
            "%H:%M"
        ).hour
    except (TypeError, ValueError):
        return "Unknown"

    if 5 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Night"


# ---------------------------------------------------------------------------
# Decision-support helper
# ---------------------------------------------------------------------------

def recommended_action(risk_level: str) -> str:
    """
    Convert a risk level into a decision-support recommendation.

    These are recommendations only and do not represent automatic
    law-enforcement commands.
    """

    actions = {
        "LOW": (
            "Continue routine monitoring and standard coverage."
        ),
        "MODERATE": (
            "Increase monitoring frequency and review recent activity."
        ),
        "HIGH": (
            "Prioritize monitoring and consider allocating additional "
            "patrol resources to the affected area."
        ),
        "CRITICAL": (
            "Recommend immediate review and prioritized resource "
            "allocation for the affected area."
        ),
    }

    return actions.get(
        str(risk_level).upper(),
        actions["LOW"]
    )