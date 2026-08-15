"""
UrbanShield - api.py
--------------------
FastAPI backend.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

API endpoints:
    GET  /
    GET  /api/health
    POST /api/incidents
    GET  /api/incidents
    GET  /api/incidents/{incident_id}
    GET  /api/risk
    GET  /api/alerts
    GET  /api/hotspots

The POST /api/incidents endpoint uses the same pipeline as
the Streamlit frontend, so manual incidents and external API
incidents are processed consistently.
"""

import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database as db
import pipeline
import hotspot_engine


# ==========================================================================
# APPLICATION
# ==========================================================================

app = FastAPI(
    title="UrbanShield API",
    description=(
        "AI-Powered Urban Crime Intelligence "
        "& Decision Support System"
    ),
    version="1.0.0",
)


# ==========================================================================
# CORS
# ==========================================================================

# Allows the existing frontend to communicate with the API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# STARTUP
# ==========================================================================

@app.on_event("startup")
def startup():
    """
    Initialize the database when the API starts.
    """

    db.init_db()


# ==========================================================================
# REQUEST MODEL
# ==========================================================================

class IncidentIn(BaseModel):
    """
    Data received when a new incident is submitted.

    This model is intentionally compatible with the
    Streamlit incident form and the processing pipeline.
    """

    crime_type: str = Field(
        ...,
        description="Type of crime"
    )

    crime_category: str = Field(
        ...,
        description="Crime category"
    )

    crime_severity: str = Field(
        ...,
        description="Infraction / Misdemeanor / Felony"
    )

    occurred_date: str = Field(
        ...,
        description="Incident date in YYYY-MM-DD format"
    )

    occurred_time: str = Field(
        ...,
        description="Incident time in HH:MM format"
    )

    district: Optional[str] = None

    neighborhood: Optional[str] = None

    latitude: float

    longitude: float

    weapon_used: str = "Unknown"

    domestic_related: str = "No"

    gang_related: str = "No"

    property_damage: str = "No"

    estimated_loss: int = 0

    priority_level: str = "Low"


# ==========================================================================
# ROOT
# ==========================================================================

@app.get("/")
def root():
    """
    Basic API status endpoint.
    """

    return {
        "service": "UrbanShield API",
        "status": "online",
        "version": "1.0.0",
    }


# ==========================================================================
# HEALTH CHECK
# ==========================================================================

@app.get("/api/health")
def health_check():
    """
    Health endpoint useful for deployment and monitoring.
    """

    try:
        db.init_db()

        return {
            "status": "healthy",
            "database": "available",
            "service": "UrbanShield API",
        }

    except Exception as e:

        return {
            "status": "degraded",
            "database": "unavailable",
            "error": str(e),
        }


# ==========================================================================
# POST NEW INCIDENT
# ==========================================================================

@app.post("/api/incidents")
def submit_incident(incident: IncidentIn):
    """
    Receive a new incident from an external system.

    Flow:

        External API
             ↓
        Validation
             ↓
        Database
             ↓
        Risk Engine
             ↓
        Anomaly Detection
             ↓
        Hotspot Detection
             ↓
        Alert
             ↓
        Decision Support
    """

    try:

        payload = incident.dict()

        result = pipeline.process_new_incident(
            payload,
            source="api"
        )

        if not result.get("success", False):

            raise HTTPException(
                status_code=422,
                detail=result.get(
                    "errors",
                    ["Incident processing failed."]
                )
            )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {e}"
        )


# ==========================================================================
# GET INCIDENTS
# ==========================================================================

@app.get("/api/incidents")
def list_incidents(
    limit: int = Query(
        default=50,
        ge=1,
        le=500
    )
):
    """
    Return recently submitted incidents.
    """

    try:

        rows = db.fetch_incidents(
            limit=limit
        )

        return rows

    except AttributeError:

        # ORM fallback

        try:

            from database import (
                get_session,
                Incident
            )

            with get_session() as session:

                rows = (
                    session.query(
                        Incident
                    )
                    .order_by(
                        Incident.submitted_at.desc()
                    )
                    .limit(limit)
                    .all()
                )

                return [
                    _incident_dict(row)
                    for row in rows
                ]

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Database error: {e}"
            )


# ==========================================================================
# GET ONE INCIDENT
# ==========================================================================

@app.get("/api/incidents/{incident_id}")
def get_incident(
    incident_id: str
):
    """
    Return one incident together with
    its risk assessment.
    """

    try:

        incident = db.fetch_incident(
            incident_id
        )

        if not incident:

            raise HTTPException(
                status_code=404,
                detail="Incident not found"
            )

        assessments = db.fetch_risk_assessments(
            incident_id=incident_id,
            limit=100
        )

        return {
            "incident": incident,
            "risk_assessments": assessments,
        }

    except HTTPException:
        raise

    except AttributeError:

        try:

            from database import (
                get_session,
                Incident,
                RiskAssessment
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

                if not row:

                    raise HTTPException(
                        status_code=404,
                        detail="Incident not found"
                    )

                assessments = (
                    session.query(
                        RiskAssessment
                    )
                    .filter_by(
                        incident_id=incident_id
                    )
                    .all()
                )

                return {
                    "incident": _incident_dict(row),
                    "risk_assessments": [
                        _risk_dict(a)
                        for a in assessments
                    ],
                }

        except HTTPException:
            raise

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Database error: {e}"
            )


# ==========================================================================
# GET RISK ASSESSMENTS
# ==========================================================================

@app.get("/api/risk")
def list_risk_assessments(
    incident_id: Optional[str] = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=500
    )
):
    """
    Return recent risk assessments.

    Optional:
        ?incident_id=INC-XXXX
    """

    try:

        return db.fetch_risk_assessments(
            incident_id=incident_id,
            limit=limit
        )

    except AttributeError:

        try:

            from database import (
                get_session,
                RiskAssessment
            )

            with get_session() as session:

                query = session.query(
                    RiskAssessment
                )

                if incident_id:

                    query = query.filter(
                        RiskAssessment.incident_id
                        == incident_id
                    )

                rows = (
                    query
                    .order_by(
                        RiskAssessment.created_at.desc()
                    )
                    .limit(limit)
                    .all()
                )

                return [
                    _risk_dict(row)
                    for row in rows
                ]

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Database error: {e}"
            )


# ==========================================================================
# GET ALERTS
# ==========================================================================

@app.get("/api/alerts")
def list_alerts(
    status: Optional[str] = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=500
    )
):
    """
    Return generated alerts.

    Example:

        /api/alerts?status=Open
    """

    try:

        return db.fetch_alerts(
            status=status,
            limit=limit
        )

    except AttributeError:

        try:

            from database import (
                get_session,
                Alert
            )

            with get_session() as session:

                query = session.query(
                    Alert
                )

                if status:

                    query = query.filter(
                        Alert.status == status
                    )

                rows = (
                    query
                    .order_by(
                        Alert.created_at.desc()
                    )
                    .limit(limit)
                    .all()
                )

                return [
                    _alert_dict(row)
                    for row in rows
                ]

        except Exception as e:

            raise HTTPException(
                status_code=500,
                detail=f"Database error: {e}"
            )


# ==========================================================================
# HOTSPOTS
# ==========================================================================

@app.get("/api/hotspots")
def get_hotspots():
    """
    Return historical and emerging hotspots.

    This endpoint powers the existing dashboard/map
    without requiring changes to its visual design.
    """

    try:

        historical = (
            hotspot_engine
            .compute_historical_hotspots()
        )

        recent_df = (
            pipeline
            ._recent_incidents_df(
                days=7
            )
        )

        emerging = (
            hotspot_engine
            .detect_emerging_hotspots(
                recent_df
            )
        )

        # Convert DataFrames to JSON-compatible records.

        if hasattr(
            historical,
            "to_dict"
        ):

            historical_records = (
                historical
                .to_dict(
                    orient="records"
                )
            )

        else:

            historical_records = historical

        if hasattr(
            emerging,
            "to_dict"
        ):

            emerging_records = (
                emerging
                .to_dict(
                    orient="records"
                )
            )

        else:

            emerging_records = emerging

        return {
            "historical_hotspots":
                historical_records,

            "emerging_hotspots":
                emerging_records,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Hotspot processing error: {e}"
        )


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def _incident_dict(
    r
) -> dict:

    """
    Convert ORM Incident object
    into a JSON-friendly dictionary.
    """

    return {

        "incident_id":
            r.incident_id,

        "crime_type":
            r.crime_type,

        "crime_category":
            r.crime_category,

        "crime_severity":
            r.crime_severity,

        "occurred_date":
            r.occurred_date,

        "occurred_time":
            r.occurred_time,

        "district":
            r.district,

        "neighborhood":
            r.neighborhood,

        "latitude":
            r.latitude,

        "longitude":
            r.longitude,

        "in_coverage":
            r.in_coverage,

        "weapon_used":
            r.weapon_used,

        "domestic_related":
            r.domestic_related,

        "gang_related":
            r.gang_related,

        "property_damage":
            r.property_damage,

        "estimated_loss":
            r.estimated_loss,

        "priority_level":
            r.priority_level,

        "source":
            r.source,

        "status":
            r.status,

        "submitted_at":
            (
                r.submitted_at.isoformat()
                if r.submitted_at
                else None
            ),
    }


def _risk_dict(
    r
) -> dict:

    """
    Convert ORM RiskAssessment
    into a JSON-friendly dictionary.
    """

    try:

        reasons = (
            json.loads(r.reasons)
            if r.reasons
            else []
        )

    except Exception:

        reasons = (
            [r.reasons]
            if r.reasons
            else []
        )

    return {

        "assessment_id":
            r.assessment_id,

        "incident_id":
            r.incident_id,

        "historical_risk":
            r.historical_risk,

        "dynamic_risk":
            r.dynamic_risk,

        "risk_level":
            r.risk_level,

        "reasons":
            reasons,

        "anomaly_flag":
            r.anomaly_flag,

        "anomaly_score":
            r.anomaly_score,

        "created_at":
            (
                r.created_at.isoformat()
                if r.created_at
                else None
            ),
    }


def _alert_dict(
    r
) -> dict:

    """
    Convert ORM Alert object
    into a JSON-friendly dictionary.
    """

    return {

        "alert_id":
            r.alert_id,

        "incident_id":
            r.incident_id,

        "location":
            r.location,

        "alert_type":
            r.alert_type,

        "risk_level":
            r.risk_level,

        "message":
            r.message,

        "recommendation":
            r.recommendation,

        "status":
            r.status,

        "created_at":
            (
                r.created_at.isoformat()
                if r.created_at
                else None
            ),
    }