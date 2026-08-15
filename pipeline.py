"""
UrbanShield - pipeline.py
-------------------------
Central processing pipeline for incoming crime incidents.

Flow:

New Incident
     ↓
Validation
     ↓
Store in Database
     ↓
Coverage Check
     ↓
Historical Risk Baseline
     ↓
Anomaly Detection
     ↓
Emerging Hotspot Detection
     ↓
Dynamic Risk Calculation
     ↓
Decision-Support Priority
     ↓
Alert Generation
     ↓
Return result to Frontend / API
"""

import pandas as pd

import database as db
import utils
import risk_engine
import anomaly_detection
import hotspot_engine


# --------------------------------------------------------------------------
# MAIN INCIDENT PROCESSING FUNCTION
# --------------------------------------------------------------------------

def process_new_incident(
    payload: dict,
    source: str = "streamlit"
):
    """
    Process one newly submitted incident.

    This function is shared by:
        - Streamlit frontend
        - FastAPI backend

    Returns a dictionary containing:
        incident
        risk assessment
        anomaly information
        hotspot information
        decision-support recommendation
        alert information
    """

    # ------------------------------------------------------------------
    # 1. LOAD HISTORICAL DATA
    # ------------------------------------------------------------------

    historical_df = _load_historical_data()

    if historical_df.empty:
        return {
            "success": False,
            "errors": ["Historical crime dataset is empty or unavailable."]
        }

    # ------------------------------------------------------------------
    # 2. VALIDATE INCIDENT
    # ------------------------------------------------------------------

    valid_crime_types = sorted(
        historical_df["crime_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    valid_districts = sorted(
        historical_df["district"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # utils.py may contain different validation implementations.
    # Use the project's validation function when available.

    try:
        ok, errors = utils.validate_incident_payload(
            payload,
            valid_crime_types,
            valid_districts
        )

        if not ok:
            return {
                "success": False,
                "errors": errors
            }

    except AttributeError:
        # Fallback validation in case utils.py uses a different interface.
        errors = _basic_validation(
            payload,
            valid_crime_types,
            valid_districts
        )

        if errors:
            return {
                "success": False,
                "errors": errors
            }

    # ------------------------------------------------------------------
    # 3. CREATE INCIDENT RECORD
    # ------------------------------------------------------------------

    incident_id = (
        payload.get("incident_id")
        or _new_incident_id()
    )

    latitude = float(payload["latitude"])
    longitude = float(payload["longitude"])

    # ------------------------------------------------------------------
    # 4. CHECK HISTORICAL GEOGRAPHIC COVERAGE
    # ------------------------------------------------------------------

    try:
        coverage_bounds = utils.compute_coverage_bounds(
            historical_df
        )

        within_coverage = utils.is_within_coverage(
            latitude,
            longitude,
            coverage_bounds
        )

    except Exception:
        # If coverage calculation is unavailable,
        # allow processing to continue.
        within_coverage = True

    # ------------------------------------------------------------------
    # 5. NORMALIZE INCIDENT RECORD
    # ------------------------------------------------------------------

    record = {
        "incident_id": incident_id,

        "crime_type": payload.get(
            "crime_type",
            "Unknown"
        ),

        "crime_category": payload.get(
            "crime_category",
            "Unknown"
        ),

        "crime_severity": payload.get(
            "crime_severity",
            "Misdemeanor"
        ),

        "district": payload.get(
            "district"
        ),

        "neighborhood": payload.get(
            "neighborhood"
        ),

        "latitude": latitude,
        "longitude": longitude,

        "occurred_date": str(
            payload.get("occurred_date", "")
        ),

        "occurred_time": payload.get(
            "occurred_time"
        ),

        "weapon_used": payload.get(
            "weapon_used",
            "Unknown"
        ),

        "domestic_related": payload.get(
            "domestic_related",
            "No"
        ),

        "gang_related": payload.get(
            "gang_related",
            "No"
        ),

        "property_damage": payload.get(
            "property_damage",
            "No"
        ),

        "estimated_loss": int(
            float(
                payload.get(
                    "estimated_loss",
                    0
                ) or 0
            )
        ),

        "priority_level": payload.get(
            "priority_level",
            "Low"
        ),

        "in_coverage": bool(
            within_coverage
        ),

        "source": source,

        "status": "Received",
    }

    # ------------------------------------------------------------------
    # 6. STORE NEW INCIDENT
    # ------------------------------------------------------------------

    stored_incident = _store_incident(record)

    # ------------------------------------------------------------------
    # 7. OUTSIDE HISTORICAL COVERAGE
    # ------------------------------------------------------------------

    if not within_coverage:

        return {
            "success": True,
            "incident": record,
            "within_coverage": False,

            "message": (
                "LIMITED HISTORICAL COVERAGE: "
                "This location is outside the geographic "
                "coverage of the historical dataset. "
                "The incident has been stored, but a "
                "location-specific historical risk score "
                "cannot be reliably calculated."
            ),

            "assessment": None,
            "anomaly": None,
            "emerging_ratio": None,
            "priority": None,
            "alert": None,
        }

    # ------------------------------------------------------------------
    # 8. HISTORICAL RISK BASELINE
    # ------------------------------------------------------------------

    baseline_df = risk_engine.compute_historical_baseline(
        historical_df
    )

    baseline_row = None

    if record["district"] is not None:

        matches = baseline_df[
            baseline_df["district"].astype(str)
            == str(record["district"])
        ]

        if not matches.empty:
            baseline_row = matches.iloc[0]

    if baseline_row is not None:
        historical_risk = float(
            baseline_row["historical_risk"]
        )
    else:
        historical_risk = 30.0

    # ------------------------------------------------------------------
    # 9. TRAIN / LOAD ANOMALY MODEL
    # ------------------------------------------------------------------

    try:

        anomaly_model, anomaly_encoders = (
            anomaly_detection.train_anomaly_model(
                historical_df
            )
        )

        (
            is_anomaly,
            anomaly_score,
            anomaly_explanation
        ) = anomaly_detection.score_incident(
            anomaly_model,
            anomaly_encoders,
            record
        )

    except Exception as e:

        is_anomaly = False
        anomaly_score = 0.0

        anomaly_explanation = (
            "Anomaly analysis unavailable: "
            + str(e)
        )

    # ------------------------------------------------------------------
    # 10. BUILD CURRENT INCIDENT DATA
    # ------------------------------------------------------------------

    try:

        recent_records = db.fetch_incidents(
            limit=2000
        )

        incoming_df = pd.DataFrame(
            recent_records
        )

    except Exception:

        incoming_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 11. DETECT EMERGING HOTSPOTS
    # ------------------------------------------------------------------

    emerging_ratio = 1.0
    emerging_hotspot = None

    try:

        emerging = (
            hotspot_engine.detect_emerging_hotspots(
                historical_df,
                incoming_df,
                window_days=14
            )
        )

        # Support DataFrame or list output.

        if isinstance(emerging, pd.DataFrame):

            if not emerging.empty:

                district_matches = emerging[
                    emerging["district"].astype(str)
                    == str(record["district"])
                ]

                if not district_matches.empty:

                    row = district_matches.iloc[0]

                    if "ratio" in row:
                        emerging_ratio = float(
                            row["ratio"]
                        )

                    emerging_hotspot = (
                        row.to_dict()
                    )

        elif isinstance(emerging, list):

            for item in emerging:

                if str(
                    item.get("district")
                ) == str(
                    record["district"]
                ):

                    emerging_ratio = float(
                        item.get(
                            "ratio",
                            1.0
                        )
                    )

                    emerging_hotspot = item

                    break

    except Exception:

        emerging_ratio = 1.0
        emerging_hotspot = None

    # ------------------------------------------------------------------
    # 12. DYNAMIC RISK CALCULATION
    # ------------------------------------------------------------------

    (
        dynamic_risk,
        risk_level,
        reasons
    ) = risk_engine.compute_dynamic_risk(
        record,
        baseline_row,
        emerging_ratio=emerging_ratio,
        is_anomaly=is_anomaly
    )

    # ------------------------------------------------------------------
    # 13. STORE RISK ASSESSMENT
    # ------------------------------------------------------------------

    assessment = {
        "incident_id": incident_id,

        "historical_risk": round(
            historical_risk,
            1
        ),

        "dynamic_risk": dynamic_risk,

        "risk_level": risk_level,

        "reasons": reasons,

        "anomaly_flag": bool(
            is_anomaly
        ),

        "anomaly_score": anomaly_score,
    }

    _store_risk_assessment(
        assessment
    )

    # ------------------------------------------------------------------
    # 14. DECISION-SUPPORT PRIORITY
    # ------------------------------------------------------------------

    priority = risk_engine.recommend_priority(
        risk_level,
        record["district"],
        reasons
    )

    # ------------------------------------------------------------------
    # 15. GENERATE ALERT
    # ------------------------------------------------------------------

    alert = None

    if risk_level in (
        "HIGH",
        "CRITICAL"
    ):

        alert_type = (
            "Risk Escalation"
            if risk_level == "CRITICAL"
            else "Elevated Risk"
        )

        alert_record = {

            "incident_id": incident_id,

            "location": (
                f"{record['district']} / "
                f"{record.get('neighborhood') or 'N/A'}"
            ),

            "alert_type": alert_type,

            "risk_level": risk_level,

            "message": (
                f"{risk_level} risk incident "
                f"({record['crime_type']}) "
                f"in {record['district']}."
            ),

            "recommendation": priority[
                "action"
            ],

            "status": "Open",
        }

        alert_id = _store_alert(
            alert_record
        )

        alert_record[
            "alert_id"
        ] = alert_id

        alert = alert_record

    # ------------------------------------------------------------------
    # 16. UPDATE INCIDENT STATUS
    # ------------------------------------------------------------------

    _mark_processed(
        incident_id
    )

    # ------------------------------------------------------------------
    # 17. FINAL RESULT
    # ------------------------------------------------------------------

    return {

        "success": True,

        "incident": record,

        "within_coverage": True,

        "assessment": assessment,

        "reasons": reasons,

        "anomaly": {

            "is_anomaly": bool(
                is_anomaly
            ),

            "score": anomaly_score,

            "explanation": (
                anomaly_explanation
            ),
        },

        "emerging_ratio": (
            round(
                emerging_ratio,
                2
            )
        ),

        "emerging_hotspot": (
            emerging_hotspot
        ),

        "priority": priority,

        "alert": alert,
    }


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================


def _load_historical_data():
    """
    Load the existing historical crime dataset.

    Your original dataset remains untouched.
    """

    possible_paths = [
        "data/crime_analysis_final_v4.csv",
        "data/crime_data.csv",
    ]

    for path in possible_paths:

        try:

            df = pd.read_csv(
                path
            )

            if not df.empty:
                return df

        except Exception:
            continue

    return pd.DataFrame()


def _new_incident_id():

    try:
        return utils.new_incident_id()

    except AttributeError:

        import uuid

        return (
            "INC-"
            + uuid.uuid4().hex[:12].upper()
        )


def _basic_validation(
    payload,
    valid_crime_types,
    valid_districts
):

    errors = []

    if not payload.get(
        "crime_type"
    ):
        errors.append(
            "Crime type is required."
        )

    if not payload.get(
        "crime_category"
    ):
        errors.append(
            "Crime category is required."
        )

    if not payload.get(
        "crime_severity"
    ):
        errors.append(
            "Crime severity is required."
        )

    if not payload.get(
        "occurred_date"
    ):
        errors.append(
            "Incident date is required."
        )

    if not payload.get(
        "occurred_time"
    ):
        errors.append(
            "Incident time is required."
        )

    if payload.get(
        "crime_type"
    ) not in valid_crime_types:

        errors.append(
            "Crime type is not present "
            "in the historical dataset."
        )

    district = payload.get(
        "district"
    )

    if district and district not in valid_districts:

        errors.append(
            "District is not present "
            "in the historical dataset."
        )

    try:

        float(
            payload.get(
                "latitude"
            )
        )

        float(
            payload.get(
                "longitude"
            )
        )

    except (
        TypeError,
        ValueError
    ):

        errors.append(
            "Latitude and longitude "
            "must be valid numbers."
        )

    return errors


def _store_incident(record):

    """
    Store incident using the database layer.

    Supports the database.py interface
    used by UrbanShield.
    """

    try:

        return db.insert_incident(
            record
        )

    except AttributeError:

        # ORM fallback

        try:

            from database import (
                get_session,
                Incident
            )

            with get_session() as session:

                row = Incident(
                    **record
                )

                session.add(
                    row
                )

                session.flush()

                return row

        except Exception as e:

            raise RuntimeError(
                f"Unable to store incident: {e}"
            )


def _store_risk_assessment(
    assessment
):

    """
    Store risk assessment.
    """

    # Convert reasons to JSON if the
    # database expects a text column.

    db_record = dict(
        assessment
    )

    if isinstance(
        db_record.get("reasons"),
        list
    ):

        import json

        db_record["reasons"] = json.dumps(
            db_record["reasons"]
        )

    try:

        return db.insert_risk_assessment(
            db_record
        )

    except AttributeError:

        try:

            from database import (
                get_session,
                RiskAssessment
            )

            import uuid

            with get_session() as session:

                row = RiskAssessment(

                    assessment_id=(
                        "ASM-"
                        + uuid.uuid4()
                        .hex[:12]
                        .upper()
                    ),

                    incident_id=(
                        db_record[
                            "incident_id"
                        ]
                    ),

                    historical_risk=(
                        db_record[
                            "historical_risk"
                        ]
                    ),

                    dynamic_risk=(
                        db_record[
                            "dynamic_risk"
                        ]
                    ),

                    risk_level=(
                        db_record[
                            "risk_level"
                        ]
                    ),

                    reasons=(
                        db_record[
                            "reasons"
                        ]
                    ),

                    anomaly_flag=(
                        db_record.get(
                            "anomaly_flag",
                            False
                        )
                    ),

                    anomaly_score=(
                        db_record.get(
                            "anomaly_score"
                        )
                    ),
                )

                session.add(
                    row
                )

                session.flush()

                return row

        except Exception as e:

            raise RuntimeError(
                f"Unable to store risk assessment: {e}"
            )


def _store_alert(
    alert_record
):

    """
    Store a generated risk alert.
    """

    try:

        return db.insert_alert(
            alert_record
        )

    except AttributeError:

        try:

            from database import (
                get_session,
                Alert
            )

            import uuid

            with get_session() as session:

                row = Alert(

                    alert_id=(
                        "ALT-"
                        + uuid.uuid4()
                        .hex[:12]
                        .upper()
                    ),

                    incident_id=(
                        alert_record.get(
                            "incident_id"
                        )
                    ),

                    location=(
                        alert_record.get(
                            "location"
                        )
                    ),

                    alert_type=(
                        alert_record.get(
                            "alert_type"
                        )
                    ),

                    risk_level=(
                        alert_record.get(
                            "risk_level"
                        )
                    ),

                    message=(
                        alert_record.get(
                            "message"
                        )
                    ),

                    recommendation=(
                        alert_record.get(
                            "recommendation"
                        )
                    ),

                    status=(
                        alert_record.get(
                            "status",
                            "Open"
                        )
                    ),
                )

                session.add(
                    row
                )

                session.flush()

                return row.alert_id

        except Exception as e:

            raise RuntimeError(
                f"Unable to store alert: {e}"
            )


def _mark_processed(
    incident_id
):

    """
    Mark incident as processed when the
    database implementation supports it.
    """

    try:

        from database import (
            get_session,
            Incident
        )

        with get_session() as session:

            row = (
                session.query(
                    Incident
                )
                .filter_by(
                    incident_id=incident_id
                )
                .first()
            )

            if row:
                row.status = "Processed"

    except Exception:
        # Status update should never cause
        # the complete incident pipeline
        # to fail.
        pass


# ==========================================================================
# RECENT INCIDENT DATA
# ==========================================================================


def _recent_incidents_df(
    days=7
):

    """
    Return recent incoming incidents as
    a DataFrame.

    Used by the hotspot API and dashboard.
    """

    try:

        rows = db.fetch_incidents(
            limit=2000
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows
        )

        if "occurred_date" in df.columns:

            df["occurred_date"] = pd.to_datetime(
                df["occurred_date"],
                errors="coerce"
            )

            cutoff = (
                pd.Timestamp.now()
                - pd.Timedelta(
                    days=days
                )
            )

            df = df[
                df["occurred_date"]
                >= cutoff
            ]

        return df

    except Exception:

        return pd.DataFrame()