from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

import database as db
from pipeline import process_new_incident


app = FastAPI(
    title="UrbanShield API",
    description="Real-time crime incident ingestion and risk analysis API",
    version="1.0.0",
)


# ============================================================
# INCIDENT REQUEST MODEL
# ============================================================

class IncidentRequest(BaseModel):
    crime_type: str
    crime_category: str
    crime_severity: str

    occurred_date: str
    occurred_time: str

    district: str
    neighborhood: Optional[str] = None

    latitude: float
    longitude: float

    weapon_used: str = "Unknown"
    domestic_related: str = "No"
    gang_related: str = "No"
    property_damage: str = "No"

    estimated_loss: float = Field(default=0, ge=0)
    priority_level: str = "Low"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "UrbanShield API",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": db.db_backend_name()
    }


# ============================================================
# REAL-TIME INCIDENT INGESTION
# ============================================================

@app.post("/api/incidents")
def receive_incident(data: IncidentRequest):

    payload = data.model_dump()

    result = process_new_incident(
        payload,
        source="api"
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get(
                "errors",
                ["Incident processing failed."]
            )
        )

    return {
        "success": True,
        "message": "Incident received and processed successfully.",
        "source": "api",
        "incident": result.get("incident"),
        "assessment": result.get("assessment"),
        "anomaly": result.get("anomaly"),
        "emerging_ratio": result.get("emerging_ratio"),
        "priority": result.get("priority"),
        "alert": result.get("alert"),
    }


# ============================================================
# DATABASE STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    return {
        "service": "UrbanShield",
        "api": "online",
        "database": db.db_backend_name(),
    }