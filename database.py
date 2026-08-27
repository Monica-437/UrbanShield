"""
UrbanShield - database.py
-------------------------

SQLAlchemy data layer for UrbanShield.

Database behavior:
    - Uses MySQL when MYSQL_* environment variables are configured.
    - Falls back to SQLite during local development when MySQL
      configuration is not available.

The ORM models in this file correspond to sql/schema.sql.

Tables:
    1. incidents
    2. risk_assessments
    3. alerts
    4. area_baselines

No database credentials are hard-coded.
All MySQL connection details come from environment variables.
"""

import os
import uuid
import pandas as pd
import datetime
import contextlib

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import QueuePool


# ============================================================
# OPTIONAL .env SUPPORT
# ============================================================

try:
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    pass


# ============================================================
# BASE
# ============================================================

Base = declarative_base()


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


# ------------------------------------------------------------
# Determine database backend
# ------------------------------------------------------------

USE_MYSQL = all(
    [
        MYSQL_HOST,
        MYSQL_USER,
        MYSQL_PASSWORD,
        MYSQL_DATABASE,
    ]
)


# ============================================================
# MYSQL DATABASE
# ============================================================

if USE_MYSQL:

    DATABASE_URL = (
        f"mysql+pymysql://"
        f"{MYSQL_USER}:{MYSQL_PASSWORD}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/"
        f"{MYSQL_DATABASE}"
        f"?charset=utf8mb4"
    )

    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


# ============================================================
# SQLITE DEVELOPMENT FALLBACK
# ============================================================

else:

    DATABASE_URL = "sqlite:///urbanshield_dev.db"

    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False
        },
        future=True,
    )


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


# ============================================================
# 1. INCIDENT MODEL
# ============================================================

class Incident(Base):

    __tablename__ = "incidents"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    incident_id = Column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
    )

    crime_type = Column(
        String(64),
        nullable=False,
    )

    crime_category = Column(
        String(32),
        nullable=False,
    )

    crime_severity = Column(
        String(32),
        nullable=False,
    )

    occurred_date = Column(
        String(16),
        nullable=False,
    )

    occurred_time = Column(
        String(16),
        nullable=False,
    )

    district = Column(
        String(64),
        nullable=True,
        index=True,
    )

    neighborhood = Column(
        String(128),
        nullable=True,
    )

    latitude = Column(
        Float,
        nullable=False,
    )

    longitude = Column(
        Float,
        nullable=False,
    )

    in_coverage = Column(
        Boolean,
        default=True,
    )

    weapon_used = Column(
        String(32),
        default="Unknown",
    )

    domestic_related = Column(
        String(8),
        default="No",
    )

    gang_related = Column(
        String(8),
        default="No",
    )

    property_damage = Column(
        String(8),
        default="No",
    )

    estimated_loss = Column(
        Integer,
        default=0,
    )

    priority_level = Column(
        String(16),
        default="Low",
    )

    source = Column(
        String(16),
        default="streamlit",
    )

    status = Column(
        String(32),
        default="Received",
    )

    submitted_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    # --------------------------------------------------------
    # Relationships
    # --------------------------------------------------------

    risk_assessments = relationship(
        "RiskAssessment",
        back_populates="incident",
        cascade="all, delete-orphan",
    )

    alerts = relationship(
        "Alert",
        back_populates="incident",
    )

    # --------------------------------------------------------
    # Additional indexes
    # --------------------------------------------------------

    __table_args__ = (
        Index(
            "idx_incidents_crime_type",
            "crime_type",
        ),

        Index(
            "idx_incidents_occurred_date",
            "occurred_date",
        ),

        Index(
            "idx_incidents_submitted",
            "submitted_at",
        ),

        Index(
            "idx_incidents_source",
            "source",
        ),

        Index(
            "idx_incidents_status",
            "status",
        ),

        Index(
            "idx_incidents_location",
            "latitude",
            "longitude",
        ),
    )


# ============================================================
# 2. RISK ASSESSMENT MODEL
# ============================================================

class RiskAssessment(Base):

    __tablename__ = "risk_assessments"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    assessment_id = Column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
    )

    incident_id = Column(
        String(32),
        ForeignKey(
            "incidents.incident_id",
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    historical_risk = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    dynamic_risk = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    risk_level = Column(
        String(16),
        nullable=False,
    )

    # JSON-encoded explanation list
    reasons = Column(
        Text,
        nullable=True,
    )

    anomaly_flag = Column(
        Boolean,
        default=False,
    )

    anomaly_score = Column(
        Float,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    incident = relationship(
        "Incident",
        back_populates="risk_assessments",
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

    __table_args__ = (
        Index(
            "idx_risk_level",
            "risk_level",
        ),

        Index(
            "idx_risk_created",
            "created_at",
        ),

        Index(
            "idx_risk_anomaly",
            "anomaly_flag",
        ),

        Index(
            "idx_risk_dynamic",
            "dynamic_risk",
        ),
    )


# ============================================================
# 3. ALERT MODEL
# ============================================================

class Alert(Base):

    __tablename__ = "alerts"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    alert_id = Column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
    )

    incident_id = Column(
        String(32),
        ForeignKey(
            "incidents.incident_id",
            onupdate="CASCADE",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    location = Column(
        String(128),
        nullable=True,
    )

    alert_type = Column(
        String(32),
        nullable=False,
    )

    risk_level = Column(
        String(16),
        nullable=False,
    )

    message = Column(
        Text,
        nullable=False,
    )

    recommendation = Column(
        Text,
        nullable=True,
    )

    status = Column(
        String(16),
        default="Open",
    )

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    incident = relationship(
        "Incident",
        back_populates="alerts",
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

    __table_args__ = (
        Index(
            "idx_alerts_status",
            "status",
        ),

        Index(
            "idx_alerts_risk_level",
            "risk_level",
        ),

        Index(
            "idx_alerts_type",
            "alert_type",
        ),

        Index(
            "idx_alerts_created",
            "created_at",
        ),
    )


# ============================================================
# 4. AREA BASELINE MODEL
# ============================================================

class AreaBaseline(Base):

    __tablename__ = "area_baselines"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    district = Column(
        String(64),
        unique=True,
        nullable=False,
    )

    historical_incident_count = Column(
        Integer,
        default=0,
    )

    avg_severity_score = Column(
        Float,
        default=0.0,
    )

    baseline_risk = Column(
        Float,
        default=0.0,
    )

    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

    __table_args__ = (
        Index(
            "idx_baseline_district",
            "district",
        ),

        Index(
            "idx_baseline_risk",
            "baseline_risk",
        ),
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Create all UrbanShield tables if they do not already exist.

    Safe to call whenever the application starts.
    """

    Base.metadata.create_all(
        bind=engine
    )
# ============================================================
# PIPELINE DATABASE HELPERS
# ============================================================

def insert_incident(record):
    """Insert one incoming incident."""
    with get_session() as session:
        incident = Incident(**record)
        session.add(incident)
        session.flush()
        session.refresh(incident)
        return incident


def fetch_incidents(limit=2000):
    """Fetch recent incidents for dynamic analysis."""
    with get_session() as session:
        rows = (
            session.query(Incident)
            .filter(Incident.source != "historical")
            .order_by(Incident.submitted_at.desc())
            .limit(limit)
            .all()
        )

        result = []

        for row in rows:
            result.append({
                "incident_id": row.incident_id,
                "crime_type": row.crime_type,
                "crime_category": row.crime_category,
                "crime_severity": row.crime_severity,
                "occurred_date": row.occurred_date,
                "occurred_time": row.occurred_time,
                "district": row.district,
                "neighborhood": row.neighborhood,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "in_coverage": row.in_coverage,
                "weapon_used": row.weapon_used,
                "domestic_related": row.domestic_related,
                "gang_related": row.gang_related,
                "property_damage": row.property_damage,
                "estimated_loss": row.estimated_loss,
                "priority_level": row.priority_level,
                "source": row.source,
                "status": row.status,
                "submitted_at": row.submitted_at,
            })

        return result


def insert_risk_assessment(record):
    """Store one risk assessment."""
    with get_session() as session:
        assessment = RiskAssessment(
            assessment_id=(
                "ASM-"
                + uuid.uuid4().hex[:12].upper()
            ),
            incident_id=record["incident_id"],
            historical_risk=record["historical_risk"],
            dynamic_risk=record["dynamic_risk"],
            risk_level=record["risk_level"],
            reasons=record["reasons"],
            anomaly_flag=record.get("anomaly_flag", False),
            anomaly_score=record.get("anomaly_score"),
        )

        session.add(assessment)
        session.flush()
        session.refresh(assessment)
        return assessment


def insert_alert(record):
    """Store one generated alert."""
    with get_session() as session:
        alert = Alert(
            alert_id=(
                "ALT-"
                + uuid.uuid4().hex[:12].upper()
            ),
            incident_id=record.get("incident_id"),
            location=record.get("location"),
            alert_type=record.get("alert_type"),
            risk_level=record.get("risk_level"),
            message=record.get("message"),
            recommendation=record.get("recommendation"),
            status=record.get("status", "Open"),
        )

        session.add(alert)
        session.flush()
        session.refresh(alert)
        return alert.alert_id


# ============================================================
# SESSION CONTEXT MANAGER
# ============================================================

@contextlib.contextmanager
def get_session():
    """
    Provide a database session with automatic:

        commit
        rollback
        close

    behavior.
    """

    session = SessionLocal()

    try:

        yield session

        session.commit()

    except Exception:

        session.rollback()

        raise

    finally:

        session.close()


# ============================================================
# DATABASE BACKEND INFORMATION
# ============================================================

def db_backend_name() -> str:
    """
    Return the currently active database backend.
    """

    if USE_MYSQL:

        return "MySQL"

    return (
        "SQLite "
        "(development fallback - "
        "set MYSQL_* environment variables for production)"
    )


# ============================================================
# DATABASE URL INFORMATION
# ============================================================

def database_url() -> str:
    """
    Return the configured database URL.

    Useful for diagnostics.

    Password is intentionally hidden from the returned value.
    """

    if USE_MYSQL:

        return (
            f"mysql+pymysql://"
            f"{MYSQL_USER}:******"
            f"@{MYSQL_HOST}:{MYSQL_PORT}/"
            f"{MYSQL_DATABASE}"
        )

    return DATABASE_URL
def import_historical_data(csv_path):
    """Import the existing 5,000 crime records as historical incidents."""

    df = pd.read_csv(csv_path)

    with get_session() as session:

        existing = (
            session.query(Incident)
            .filter(Incident.source == "historical")
            .count()
        )

        if existing > 0:
            return existing

        imported = 0

        for index, row in df.iterrows():

            def val(column, default=None):
                value = row.get(column, default)

                if pd.isna(value):
                    return default

                return value

            incident = Incident(
                incident_id=str(
                    val("incident_id", f"HIST-{index + 1:05d}")
                ),

                crime_type=str(
                    val("crime_type", "Unknown")
                ),

                crime_category=val("crime_category"),

                crime_severity=val("crime_severity"),

                occurred_date=val("occurred_date"),

                occurred_time=val("occurred_time"),

                district=val("district"),

                neighborhood=val("neighborhood"),

                latitude=val("latitude"),

                longitude=val("longitude"),

                in_coverage=True,

                weapon_used=val("weapon_used"),

                domestic_related=val("domestic_related"),

                gang_related=val("gang_related"),

                property_damage=val("property_damage"),

                estimated_loss=val("estimated_loss"),

                priority_level=val("priority_level"),

                source="historical",

                status="Historical"
            )

            session.add(incident)
            imported += 1

        session.flush()

        return imported