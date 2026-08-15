"""
UrbanShield - hotspot_engine.py
--------------------------------

Historical and emerging crime hotspot analysis.

This module provides two complementary views:

1. HISTORICAL HOTSPOTS
   Finds districts/areas that have historically experienced
   comparatively high crime activity.

2. EMERGING HOTSPOTS
   Compares recent incoming activity with the historical baseline
   to identify areas where crime activity is increasing.

The output is used by:
    - Dashboard
    - Risk engine
    - Early-warning alerts
    - Decision-support/resource allocation

Important:
    Hotspot results are analytical indicators only. They should not
    be interpreted as proof that an area or person is inherently
    dangerous.
"""

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEFAULT_RECENT_DAYS = 7
DEFAULT_HISTORICAL_DAYS = 30

EMERGING_RATIO_THRESHOLD = 1.5

HOTSPOT_LEVELS = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# -------------------------------------------------------------------
# Optional Streamlit cache helper
# -------------------------------------------------------------------

def _cache_data(func):
    """
    Use Streamlit caching when available.

    The module can still run from FastAPI or normal Python without
    requiring Streamlit.
    """
    if st is not None:
        return st.cache_data(show_spinner=False)(func)

    return func


# -------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------

def _ensure_date_column(
    df: pd.DataFrame,
    column: str = "occurred_date",
) -> pd.DataFrame:
    """
    Return a copy with a normalized datetime column.
    """

    data = df.copy()

    if column not in data.columns:
        data[column] = pd.NaT

    data[column] = pd.to_datetime(
        data[column],
        errors="coerce",
    )

    return data


def _find_location_column(df: pd.DataFrame):
    """
    Determine the best available geographic grouping column.
    """

    for column in [
        "district",
        "neighborhood",
        "location",
    ]:
        if column in df.columns:
            return column

    return None


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    """
    Convert a pandas series to numeric safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(default)


# -------------------------------------------------------------------
# Historical hotspot analysis
# -------------------------------------------------------------------

@_cache_data
def compute_historical_hotspots(
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical crime hotspots from the historical dataset.

    The score is based primarily on:
        - incident volume
        - severity
        - weapon involvement
        - domestic/gang-related activity

    Returns a dataframe containing:
        district/location
        incident_count
        felony_count
        felony_rate
        weapon_rate
        average_loss
        hotspot_score
        hotspot_level
    """

    if historical_df is None or historical_df.empty:
        return pd.DataFrame(
            columns=[
                "district",
                "incident_count",
                "felony_count",
                "felony_rate",
                "weapon_rate",
                "average_loss",
                "hotspot_score",
                "hotspot_level",
            ]
        )

    df = historical_df.copy()

    location_column = _find_location_column(df)

    if location_column is None:
        return pd.DataFrame()

    # Ensure expected columns exist.
    if "crime_severity" not in df.columns:
        df["crime_severity"] = "Unknown"

    if "weapon_used" not in df.columns:
        df["weapon_used"] = "Unknown"

    if "estimated_loss" not in df.columns:
        df["estimated_loss"] = 0

    grouped = df.groupby(
        location_column,
        dropna=False,
    )

    result = pd.DataFrame({
        "incident_count": grouped.size(),

        "felony_count": grouped[
            "crime_severity"
        ].apply(
            lambda s: (
                s.astype(str).str.lower() == "felony"
            ).sum()
        ),

        "felony_rate": grouped[
            "crime_severity"
        ].apply(
            lambda s: (
                s.astype(str).str.lower() == "felony"
            ).mean()
        ),

        "weapon_rate": grouped[
            "weapon_used"
        ].apply(
            lambda s: (
                ~s.astype(str).str.lower().isin(
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

        "average_loss": grouped[
            "estimated_loss"
        ].mean(),
    }).reset_index()

    result = result.rename(
        columns={
            location_column: "district"
        }
    )

    # ---------------------------------------------------------------
    # Normalize each component between 0 and 1
    # ---------------------------------------------------------------

    def normalize(series):
        series = _safe_numeric(series)

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

    result["n_incidents"] = normalize(
        result["incident_count"]
    )

    result["n_felony"] = normalize(
        result["felony_rate"]
    )

    result["n_weapon"] = normalize(
        result["weapon_rate"]
    )

    result["n_loss"] = normalize(
        result["average_loss"]
    )

    # ---------------------------------------------------------------
    # Hotspot score
    # ---------------------------------------------------------------
    #
    # Incident frequency receives the largest contribution because
    # hotspot analysis primarily concerns concentration of activity.
    #
    # Severity and weapon involvement provide additional context.
    #

    result["hotspot_score"] = (
        result["n_incidents"] * 50
        + result["n_felony"] * 20
        + result["n_weapon"] * 15
        + result["n_loss"] * 15
    ).clip(0, 100)

    result["hotspot_score"] = (
        result["hotspot_score"]
        .round(1)
    )

    # ---------------------------------------------------------------
    # Categorize hotspot level
    # ---------------------------------------------------------------

    result["hotspot_level"] = result[
        "hotspot_score"
    ].apply(
        _hotspot_level_from_score
    )

    return result[
        [
            "district",
            "incident_count",
            "felony_count",
            "felony_rate",
            "weapon_rate",
            "average_loss",
            "hotspot_score",
            "hotspot_level",
        ]
    ].sort_values(
        "hotspot_score",
        ascending=False,
    ).reset_index(drop=True)


# -------------------------------------------------------------------
# Hotspot level
# -------------------------------------------------------------------

def _hotspot_level_from_score(
    score: float,
) -> str:
    """
    Convert a 0-100 hotspot score into a descriptive level.
    """

    score = float(score)

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# -------------------------------------------------------------------
# Emerging hotspot detection
# -------------------------------------------------------------------

def detect_emerging_hotspots(
    historical_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    window_days: int = DEFAULT_RECENT_DAYS,
) -> list:
    """
    Detect areas where recent incoming incident activity is higher
    than the historical baseline.

    Comparison:

        recent incident rate
                 /
        historical incident rate

    Example:

        Historical = 10 incidents/week
        Recent     = 20 incidents/week

        Ratio = 2.0x

    A ratio above EMERGING_RATIO_THRESHOLD is treated as an
    emerging activity signal.

    Returns a list of dictionaries so it can be directly consumed
    by the API and pipeline.
    """

    if historical_df is None or historical_df.empty:
        return []

    if incoming_df is None or incoming_df.empty:
        return []

    historical = _ensure_date_column(
        historical_df
    )

    incoming = _ensure_date_column(
        incoming_df
    )

    location_column = _find_location_column(
        historical
    )

    if location_column is None:
        return []

    if location_column not in incoming.columns:
        return []

    # ---------------------------------------------------------------
    # Determine the recent time window
    # ---------------------------------------------------------------

    valid_dates = incoming[
        "occurred_date"
    ].dropna()

    if valid_dates.empty:
        return []

    latest_date = valid_dates.max()

    recent_start = (
        latest_date
        - pd.Timedelta(days=window_days - 1)
    )

    recent = incoming[
        incoming["occurred_date"]
        >= recent_start
    ].copy()

    if recent.empty:
        return []

    # ---------------------------------------------------------------
    # Historical time range
    # ---------------------------------------------------------------

    historical_dates = historical[
        "occurred_date"
    ].dropna()

    if historical_dates.empty:
        return []

    historical_start = historical_dates.min()
    historical_end = historical_dates.max()

    historical_span_days = max(
        (
            historical_end
            - historical_start
        ).days + 1,
        1,
    )

    # Historical incidents per day.
    historical_counts = (
        historical.groupby(location_column)
        .size()
    )

    historical_daily_rate = (
        historical_counts
        / historical_span_days
    )

    # ---------------------------------------------------------------
    # Recent counts
    # ---------------------------------------------------------------

    recent_counts = (
        recent.groupby(location_column)
        .size()
    )

    recent_span_days = max(
        window_days,
        1,
    )

    recent_daily_rate = (
        recent_counts
        / recent_span_days
    )

    # ---------------------------------------------------------------
    # Compare recent and historical activity
    # ---------------------------------------------------------------

    all_locations = sorted(
        set(
            historical_daily_rate.index
        )
        | set(
            recent_daily_rate.index
        )
    )

    results = []

    for location in all_locations:

        historical_rate = float(
            historical_daily_rate.get(
                location,
                0.0,
            )
        )

        recent_rate = float(
            recent_daily_rate.get(
                location,
                0.0,
            )
        )

        # Avoid division by zero.
        if historical_rate <= 0:
            if recent_rate > 0:
                ratio = float("inf")
            else:
                ratio = 1.0
        else:
            ratio = (
                recent_rate
                / historical_rate
            )

        # -----------------------------------------------------------
        # Determine whether activity is emerging.
        # -----------------------------------------------------------

        is_emerging = (
            ratio >= EMERGING_RATIO_THRESHOLD
            and recent_rate > 0
        )

        # -----------------------------------------------------------
        # Human-readable trend
        # -----------------------------------------------------------

        if ratio > 1.05:
            trend = "Increasing"
        elif ratio < 0.95:
            trend = "Decreasing"
        else:
            trend = "Stable"

        results.append({
            "district": location,
            "recent_incidents": int(
                recent_counts.get(
                    location,
                    0,
                )
            ),
            "historical_daily_rate": round(
                historical_rate,
                3,
            ),
            "recent_daily_rate": round(
                recent_rate,
                3,
            ),
            "ratio": (
                round(ratio, 2)
                if np.isfinite(ratio)
                else 99.0
            ),
            "trend": trend,
            "is_emerging": bool(
                is_emerging
            ),
        })

    # Highest increase first.
    results.sort(
        key=lambda x: x["ratio"],
        reverse=True,
    )

    return results


# -------------------------------------------------------------------
# Early-warning alerts
# -------------------------------------------------------------------

def generate_early_warning(
    emerging_hotspots: list,
) -> list:
    """
    Convert emerging hotspot results into dashboard-ready alerts.

    Only areas that satisfy the emerging hotspot threshold are
    included.
    """

    warnings = []

    for item in emerging_hotspots:

        if not item.get(
            "is_emerging",
            False,
        ):
            continue

        district = item.get(
            "district",
            "Unknown",
        )

        ratio = float(
            item.get(
                "ratio",
                1.0,
            )
        )

        recent_count = int(
            item.get(
                "recent_incidents",
                0,
            )
        )

        if ratio >= 3:
            severity = "CRITICAL"
        elif ratio >= 2:
            severity = "HIGH"
        else:
            severity = "MODERATE"

        warnings.append({
            "district": district,
            "alert_type": "Emerging Crime Activity",
            "severity": severity,
            "ratio": ratio,
            "recent_incidents": recent_count,
            "message": (
                f"{district} has experienced an unusual increase "
                f"in recent incident activity "
                f"({ratio:.1f}x the historical baseline rate)."
            ),
            "recommendation": (
                "Increase monitoring frequency and review "
                "available patrol resources for this area."
            ),
        })

    return warnings


# -------------------------------------------------------------------
# Combined hotspot analysis
# -------------------------------------------------------------------

def build_hotspot_summary(
    historical_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    window_days: int = DEFAULT_RECENT_DAYS,
) -> dict:
    """
    Generate a complete hotspot summary.

    This function is useful for dashboard/API integration.
    """

    historical = compute_historical_hotspots(
        historical_df
    )

    emerging = detect_emerging_hotspots(
        historical_df,
        incoming_df,
        window_days=window_days,
    )

    warnings = generate_early_warning(
        emerging
    )

    return {
        "historical_hotspots": historical.to_dict(
            orient="records"
        ),
        "emerging_hotspots": emerging,
        "early_warnings": warnings,
    }


# -------------------------------------------------------------------
# District risk lookup
# -------------------------------------------------------------------

def get_hotspot_for_district(
    historical_df: pd.DataFrame,
    district: str,
):
    """
    Retrieve historical hotspot information for one district.

    Returns None if the district is not found.
    """

    hotspots = compute_historical_hotspots(
        historical_df
    )

    if hotspots.empty:
        return None

    match = hotspots[
        hotspots["district"].astype(str)
        == str(district)
    ]

    if match.empty:
        return None

    return match.iloc[0].to_dict()


# -------------------------------------------------------------------
# Resource-allocation support
# -------------------------------------------------------------------

def calculate_hotspot_priority(
    historical_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce a ranked decision-support list for areas.

    This does NOT automatically command or dispatch resources.

    It simply provides a ranked analytical basis that the dashboard
    can use when recommending allocation of a limited number of
    patrol/monitoring resources.
    """

    historical = compute_historical_hotspots(
        historical_df
    )

    emerging = detect_emerging_hotspots(
        historical_df,
        incoming_df,
    )

    if historical.empty:
        return pd.DataFrame()

    emerging_df = pd.DataFrame(
        emerging
    )

    if emerging_df.empty:
        historical["emerging_ratio"] = 1.0
    else:
        emerging_df = emerging_df[
            [
                "district",
                "ratio",
            ]
        ].rename(
            columns={
                "ratio": "emerging_ratio"
            }
        )

        historical = historical.merge(
            emerging_df,
            on="district",
            how="left",
        )

        historical[
            "emerging_ratio"
        ] = historical[
            "emerging_ratio"
        ].fillna(1.0)

    # ---------------------------------------------------------------
    # Resource priority score
    # ---------------------------------------------------------------

    historical[
        "resource_priority_score"
    ] = (
        historical["hotspot_score"] * 0.7
        + (
            historical["emerging_ratio"]
            .clip(upper=3.0)
            / 3.0
            * 30
        )
    ).clip(0, 100)

    historical[
        "resource_priority_score"
    ] = historical[
        "resource_priority_score"
    ].round(1)

    return historical.sort_values(
        "resource_priority_score",
        ascending=False,
    ).reset_index(drop=True)