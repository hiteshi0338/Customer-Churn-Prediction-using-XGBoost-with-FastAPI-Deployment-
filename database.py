from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite database file — created automatically on first run
DATABASE_URL = "sqlite:///./churn_predictions.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── Table schema ──────────────────────────────────────────
class PredictionLog(Base):
    __tablename__ = "predictions"

    id                 = Column(Integer, primary_key=True, index=True)
    timestamp          = Column(DateTime, default=datetime.utcnow)

    # Input features
    gender             = Column(String)
    senior_citizen     = Column(Integer)
    partner            = Column(String)
    dependents         = Column(String)
    tenure             = Column(Integer)
    phone_service      = Column(String)
    paperless_billing  = Column(String)
    monthly_charges    = Column(Float)
    total_charges      = Column(Float)
    contract           = Column(String)
    payment_method     = Column(String)
    internet_service   = Column(String)

    # Prediction outputs
    churn_probability  = Column(Float)
    prediction         = Column(String)
    risk_level         = Column(String)

    # Top SHAP factors (stored as a string)
    top_factors        = Column(String)


def create_tables():
    Base.metadata.create_all(bind=engine)