"""
UrbanShield - risk_engine.py
----------------------------

Historical risk baseline + dynamic incident risk scoring +
explainability + decision-support recommendations +
patrol resource allocation.

Risk levels:
    0  - 34.9   -> LOW
    35 - 54.9   -> MODERATE
    55 - 74.9   -> HIGH
    75 - 100    -> CRITICAL

IMPORTANT:
These scores are decision-support indicators only.
They are NOT automatic law-enforcement commands.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# RISK THRESHOLDS
# ============================================================

RISK_THRESHOLDS = {
    "LOW": 35,
    "MODERATE": 55,
    "HIGH": 75,
}


# ============================================================
# INCIDENT SEVERITY WEIGHTS
# ============================================================

SEVERITY_WEIGHT = {
    "Infraction": 0.20,
    "Misdemeanor": 0.50,
    "Felony": 1.00,

    # Useful when the incoming frontend/API sends these values.
    "Low": 0.20,
    "Medium": 0.50,
    "High": 0.80,
    "Critical": 1.00,
}


# ============================================================
# RESOURCE ALLOCATION SETTINGS
# ============================================================

DEFAULT_PATROL_UNITS = 10


RESOURCE_WEIGHT = {
    "CRITICAL": 4.0,
    "HIGH": 3.0,
    "MODERATE": 2.0,
    "LOW": 1.0,
}


# ============================================================
# HISTORICAL BASELINE
# ============================================================

def compute_historical_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a data-driven historical risk baseline for every district.

    Historical signals:

        1. Incident frequency
        2. Felony proportion
        3. Weapon involvement
        4. Domestic-related incidents
        5. Gang-related incidents
        6. Average estimated loss

    Every signal is normalized across districts.

    Final score:
        0 - 100
    """

    if df is None or df.empty:

        return pd.DataFrame(
            columns=[
                "district",
                "incident_count",
                "daily_rate",
                "felony_share",
                "weapon_rate",
                "domestic_rate",
                "gang_rate",
                "avg_loss",
                "historical_risk",
            ]
        )

    data = df.copy()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "district",
        "occurred_date",
        "crime_severity",
        "weapon_used",
        "domestic_related",
        "gang_related",
        "estimated_loss",
    ]

    for column in required_columns:

        if column not in data.columns:

            if column == "estimated_loss":
                data[column] = 0

            elif column in [
                "weapon_used",
                "domestic_related",
                "gang_related",
            ]:
                data[column] = "Unknown"

            else:
                raise ValueError(
                    f"Historical dataset is missing required column: {column}"
                )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    data["occurred_date"] = pd.to_datetime(
        data["occurred_date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["occurred_date"]
    )

    if data.empty:

        return pd.DataFrame(
            columns=[
                "district",
                "incident_count",
                "daily_rate",
                "felony_share",
                "weapon_rate",
                "domestic_rate",
                "gang_rate",
                "avg_loss",
                "historical_risk",
            ]
        )

    # --------------------------------------------------------
    # Dataset time span
    # --------------------------------------------------------

    span_days = max(
        (
            data["occurred_date"].max()
            - data["occurred_date"].min()
        ).days,
        1,
    )

    # --------------------------------------------------------
    # Group by district
    # --------------------------------------------------------

    grouped = data.groupby(
        "district",
        dropna=False,
    )

    stats = pd.DataFrame(
        {
            "incident_count": grouped.size(),

            "daily_rate": (
                grouped.size()
                / span_days
            ),

            "felony_share": grouped[
                "crime_severity"
            ].apply(
                lambda s:
                (
                    s.astype(str).str.lower()
                    == "felony"
                ).mean()
            ),

            "weapon_rate": grouped[
                "weapon_used"
            ].apply(
                lambda s:
                (
                    ~s.fillna("Unknown")
                    .astype(str)
                    .str.lower()
                    .isin(
                        [
                            "unknown",
                            "none",
                            "no",
                            "",
                            "nan",
                        ]
                    )
                ).mean()
            ),

            "domestic_rate": grouped[
                "domestic_related"
            ].apply(
                lambda s:
                (
                    s.astype(str).str.lower()
                    == "yes"
                ).mean()
            ),

            "gang_rate": grouped[
                "gang_related"
            ].apply(
                lambda s:
                (
                    s.astype(str).str.lower()
                    == "yes"
                ).mean()
            ),

            "avg_loss": grouped[
                "estimated_loss"
            ].mean(),
        }
    ).reset_index()

    # --------------------------------------------------------
    # Normalize helper
    # --------------------------------------------------------

    def normalize(series: pd.Series) -> pd.Series:

        series = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0)

        low = series.min()
        high = series.max()

        if high - low < 1e-9:

            return pd.Series(
                0.5,
                index=series.index,
            )

        return (
            (series - low)
            / (high - low)
        )

    # --------------------------------------------------------
    # Normalize all risk signals
    # --------------------------------------------------------

    stats["n_freq"] = normalize(
        stats["daily_rate"]
    )

    stats["n_felony"] = normalize(
        stats["felony_share"]
    )

    stats["n_weapon"] = normalize(
        stats["weapon_rate"]
    )

    stats["n_domestic"] = normalize(
        stats["domestic_rate"]
    )

    stats["n_gang"] = normalize(
        stats["gang_rate"]
    )

    stats["n_loss"] = normalize(
        stats["avg_loss"]
    )

    # --------------------------------------------------------
    # Weighted historical risk
    # --------------------------------------------------------

    stats["historical_risk"] = (
        stats["n_freq"] * 35
        + stats["n_felony"] * 25
        + stats["n_weapon"] * 15
        + stats["n_domestic"] * 10
        + stats["n_gang"] * 10
        + stats["n_loss"] * 5
    ).clip(
        0,
        100,
    ).round(1)

    return (
        stats[
            [
                "district",
                "incident_count",
                "daily_rate",
                "felony_share",
                "weapon_rate",
                "domestic_rate",
                "gang_rate",
                "avg_loss",
                "historical_risk",
            ]
        ]
        .sort_values(
            "historical_risk",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level_from_score(
    score: float,
) -> str:
    """
    Convert a 0-100 risk score into a risk category.
    """

    score = float(score)

    if score < RISK_THRESHOLDS["LOW"]:
        return "LOW"

    if score < RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"

    if score < RISK_THRESHOLDS["HIGH"]:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# RISK LEVEL COLOR / DISPLAY INFORMATION
# ============================================================

def risk_display_info(
    risk_level: str,
) -> dict:
    """
    Return presentation information for the frontend.

    This does not control the frontend.
    It simply gives app.py consistent labels.
    """

    mapping = {
        "LOW": {
            "label": "LOW",
            "icon": "🟢",
            "description": "Routine monitoring recommended.",
        },

        "MODERATE": {
            "label": "MODERATE",
            "icon": "🟡",
            "description": "Increased monitoring recommended.",
        },

        "HIGH": {
            "label": "HIGH",
            "icon": "🟠",
            "description": "Patrol priority should be considered.",
        },

        "CRITICAL": {
            "label": "CRITICAL",
            "icon": "🔴",
            "description": "Immediate review and prioritized response allocation recommended.",
        },
    }

    return mapping.get(
        risk_level,
        mapping["LOW"],
    )


# ============================================================
# DYNAMIC RISK
# ============================================================

def compute_dynamic_risk(
    incident: dict,
    historical_baseline_row=None,
    emerging_ratio: float = 1.0,
    is_anomaly: bool = False,
    anomaly_score: Optional[float] = None,
):
    """
    Calculate incident-specific dynamic risk.

    The final score combines:

        Historical district baseline
        +
        Incident severity
        +
        Weapon involvement
        +
        Domestic/gang indicators
        +
        Recent activity trend
        +
        Anomaly detection

    Returns:

        dynamic_risk
        risk_level
        reasons
    """

    reasons = []

    # --------------------------------------------------------
    # Historical baseline
    # --------------------------------------------------------

    if historical_baseline_row is not None:

        try:

            base = float(
                historical_baseline_row[
                    "historical_risk"
                ]
            )

        except Exception:

            base = 30.0

    else:

        base = 30.0

    reasons.append(
        f"Historical district baseline: {base:.1f}/100."
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity = str(
        incident.get(
            "crime_severity",
            "Misdemeanor",
        )
    )

    severity_weight = SEVERITY_WEIGHT.get(
        severity,
        0.50,
    )

    severity_boost = (
        severity_weight * 15
    )

    if severity_weight >= 0.8:

        reasons.append(
            f"Incident severity ({severity}) increases the risk score."
        )

    elif severity_weight >= 0.5:

        reasons.append(
            f"Incident severity ({severity}) contributes to the risk score."
        )

    # --------------------------------------------------------
    # Weapon involvement
    # --------------------------------------------------------

    weapon = incident.get(
        "weapon_used",
        "Unknown",
    )

    weapon_text = str(
        weapon
    ).strip().lower()

    weapon_involved = (
        weapon_text not in {
            "",
            "unknown",
            "none",
            "no",
            "nan",
            "false",
        }
    )

    weapon_boost = (
        10
        if weapon_involved
        else 0
    )

    if weapon_involved:

        reasons.append(
            f"Weapon involvement detected ({weapon})."
        )

    # --------------------------------------------------------
    # Domestic-related incidents
    # --------------------------------------------------------

    domestic = str(
        incident.get(
            "domestic_related",
            "No",
        )
    ).lower()

    domestic_boost = (
        4
        if domestic == "yes"
        else 0
    )

    if domestic == "yes":

        reasons.append(
            "Incident is domestic-related."
        )

    # --------------------------------------------------------
    # Gang-related incidents
    # --------------------------------------------------------

    gang = str(
        incident.get(
            "gang_related",
            "No",
        )
    ).lower()

    gang_boost = (
        4
        if gang == "yes"
        else 0
    )

    if gang == "yes":

        reasons.append(
            "Incident is gang-related."
        )

    # --------------------------------------------------------
    # Emerging trend
    # --------------------------------------------------------

    try:

        emerging_ratio = float(
            emerging_ratio
        )

    except Exception:

        emerging_ratio = 1.0

    trend_boost = 0.0

    if emerging_ratio > 1.0:

        trend_boost = min(
            (emerging_ratio - 1.0) * 20,
            30,
        )

        reasons.append(
            f"Recent incident activity is {emerging_ratio:.1f}x the historical baseline rate."
        )

    # --------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------

    anomaly_boost = (
        12
        if is_anomaly
        else 0
    )

    if is_anomaly:

        if anomaly_score is not None:

            reasons.append(
                f"Incident profile was flagged as anomalous (score: {float(anomaly_score):.3f})."
            )

        else:

            reasons.append(
                "Incident profile was flagged as statistically anomalous."
            )

    # --------------------------------------------------------
    # Final dynamic score
    # --------------------------------------------------------

    dynamic = (
        base * 0.50
        + severity_boost
        + weapon_boost
        + domestic_boost
        + gang_boost
        + trend_boost
        + anomaly_boost
    )

    dynamic = max(
        0.0,
        min(
            100.0,
            dynamic,
        ),
    )

    dynamic = round(
        dynamic,
        1,
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    level = risk_level_from_score(
        dynamic
    )

    reasons.append(
        f"Combined dynamic risk score: {dynamic:.1f}/100 → {level}."
    )

    return (
        dynamic,
        level,
        reasons,
    )


# ============================================================
# DECISION SUPPORT PRIORITY
# ============================================================

PRIORITY_MAP = {

    "LOW": (
        "Routine Monitoring",
        "Continue standard monitoring; no immediate escalation is indicated.",
    ),

    "MODERATE": (
        "Increased Monitoring",
        "Increase monitoring frequency in this area and review recent activity.",
    ),

    "HIGH": (
        "Patrol Priority",
        "Consider elevating patrol priority and situational awareness in this area.",
    ),

    "CRITICAL": (
        "Emergency Response Priority",
        "Recommend immediate review and prioritized response resource allocation.",
    ),
}


def recommend_priority(
    risk_level: str,
    district: str,
    reasons: list[str],
):
    """
    Convert a risk level into an actionable decision-support
    recommendation.

    IMPORTANT:
    This is a recommendation for human review.
    It is not an automatic law-enforcement command.
    """

    focus, action = PRIORITY_MAP.get(
        risk_level,
        PRIORITY_MAP["LOW"],
    )

    return {
        "area": district,
        "priority": risk_level,
        "recommended_focus": focus,
        "action": action,
        "reason": (
            reasons[-2]
            if len(reasons) > 1
            else (
                reasons[0]
                if reasons
                else "Based on the current risk assessment."
            )
        ),
        "decision_support_only": True,
    }


# ============================================================
# RESOURCE ALLOCATION
# ============================================================

def allocate_patrol_resources(
    risk_by_area,
    total_units: int = DEFAULT_PATROL_UNITS,
):
    """
    Recommend patrol-resource distribution across areas.

    Example:

        Zone 4 -> HIGH
        Zone 2 -> HIGH
        Zone 1 -> MODERATE
        Zone 3 -> LOW

    With 10 units, the function produces a proportional
    recommendation based on risk priority.

    IMPORTANT:
    This is decision support only.
    It does not automatically dispatch officers or patrols.
    """

    total_units = max(
        int(total_units),
        1,
    )

    if risk_by_area is None:

        return []

    # --------------------------------------------------------
    # Accept DataFrame or list of dictionaries
    # --------------------------------------------------------

    if isinstance(
        risk_by_area,
        pd.DataFrame,
    ):

        records = risk_by_area.to_dict(
            orient="records"
        )

    elif isinstance(
        risk_by_area,
        list,
    ):

        records = list(
            risk_by_area
        )

    else:

        raise TypeError(
            "risk_by_area must be a DataFrame or list of dictionaries."
        )

    if not records:

        return []

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    cleaned = []

    for row in records:

        area = (
            row.get("district")
            or row.get("area")
            or row.get("zone")
            or "Unknown"
        )

        level = str(
            row.get(
                "risk_level",
                "LOW",
            )
        ).upper()

        score = row.get(
            "dynamic_risk",
            row.get(
                "risk_score",
                0,
            ),
        )

        try:
            score = float(score)
        except Exception:
            score = 0.0

        weight = RESOURCE_WEIGHT.get(
            level,
            1.0,
        )

        cleaned.append(
            {
                "area": area,
                "risk_level": level,
                "risk_score": score,
                "weight": weight,
            }
        )

    # --------------------------------------------------------
    # Calculate weighted allocation
    # --------------------------------------------------------

    total_weight = sum(
        row["weight"]
        for row in cleaned
    )

    if total_weight <= 0:

        total_weight = float(
            len(cleaned)
        )

        for row in cleaned:
            row["weight"] = 1.0

    for row in cleaned:

        exact_units = (
            row["weight"]
            / total_weight
            * total_units
        )

        row["exact_units"] = exact_units

        row["recommended_units"] = int(
            np.floor(
                exact_units
            )
        )

    # --------------------------------------------------------
    # Ensure every area receives at least one unit when possible
    # --------------------------------------------------------

    if total_units >= len(cleaned):

        for row in cleaned:

            if (
                row["recommended_units"]
                < 1
            ):

                row["recommended_units"] = 1

    # --------------------------------------------------------
    # Correct rounding difference
    # --------------------------------------------------------

    allocated = sum(
        row["recommended_units"]
        for row in cleaned
    )

    remaining = (
        total_units
        - allocated
    )

    # Highest fractional remainders first
    if remaining > 0:

        cleaned.sort(
            key=lambda row:
            (
                row["exact_units"]
                - row["recommended_units"]
            ),
            reverse=True,
        )

        index = 0

        while remaining > 0:

            cleaned[
                index
            ]["recommended_units"] += 1

            remaining -= 1

            index = (
                index + 1
            ) % len(cleaned)

    # Too many units due to minimum-one rule
    elif remaining < 0:

        cleaned.sort(
            key=lambda row:
            (
                row["exact_units"]
                - row["recommended_units"]
            )
        )

        index = 0

        while remaining < 0:

            if (
                cleaned[index][
                    "recommended_units"
                ] > 1
            ):

                cleaned[index][
                    "recommended_units"
                ] -= 1

                remaining += 1

            index = (
                index + 1
            ) % len(cleaned)

    # --------------------------------------------------------
    # Restore highest-risk-first ordering
    # --------------------------------------------------------

    cleaned.sort(
        key=lambda row: (
            RESOURCE_WEIGHT.get(
                row["risk_level"],
                1,
            ),
            row["risk_score"],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    result = []

    for row in cleaned:

        result.append(
            {
                "area": row["area"],
                "risk_level": row["risk_level"],
                "risk_score": round(
                    row["risk_score"],
                    1,
                ),
                "recommended_units": row[
                    "recommended_units"
                ],
                "decision_support_only": True,
            }
        )

    return result


# ============================================================
# HIGH-RISK AREA EXTRACTION
# ============================================================

def get_priority_areas(
    risk_by_area,
    minimum_level: str = "HIGH",
):
    """
    Return areas at or above the requested risk level.
    """

    level_order = {
        "LOW": 1,
        "MODERATE": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    minimum_value = level_order.get(
        minimum_level.upper(),
        3,
    )

    if isinstance(
        risk_by_area,
        pd.DataFrame,
    ):

        records = risk_by_area.to_dict(
            orient="records"
        )

    else:

        records = risk_by_area or []

    result = []

    for row in records:

        level = str(
            row.get(
                "risk_level",
                "LOW",
            )
        ).upper()

        if (
            level_order.get(
                level,
                1,
            )
            >= minimum_value
        ):

            result.append(
                row
            )

    return result


# ============================================================
# RISK SUMMARY
# ============================================================

def summarize_risk(
    score: float,
) -> dict:
    """
    Return a compact risk summary for dashboard cards.
    """

    score = round(
        max(
            0.0,
            min(
                100.0,
                float(score),
            ),
        ),
        1,
    )

    level = risk_level_from_score(
        score
    )

    display = risk_display_info(
        level
    )

    return {
        "score": score,
        "risk_level": level,
        "label": display["label"],
        "icon": display["icon"],
        "description": display[
            "description"
        ],
    }