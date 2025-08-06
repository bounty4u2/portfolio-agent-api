"""
AlphaSheet Intelligence™ - Database Models
PostgreSQL database setup for production
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from Railway environment
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///alphasheet.db')

# Fix for SQLAlchemy (Railway provides postgresql://, SQLAlchemy needs postgresql+psycopg2://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Customer(Base):
    """Customer model"""
    __tablename__ = 'customers'
    
    customer_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    tier = Column(String, default='starter')
    region = Column(String, default='US')
    stripe_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UsageTracking(Base):
    """Usage tracking model"""
    __tablename__ = 'usage_tracking'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(String, nullable=False)
    month = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    reports_generated = Column(Integer, default=0)
    last_report_time = Column(DateTime, nullable=True)
    portfolio_count = Column(Integer, default=0)

class Report(Base):
    """Report history model"""
    __tablename__ = 'reports'
    
    report_id = Column(String, primary_key=True)
    customer_id = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)
    portfolio_value = Column(Float)
    report_type = Column(String)
    tier_used = Column(String)

# Database initialization
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database operations
class DatabaseOperations:
    """Database helper functions"""
    
    @staticmethod
    def create_customer(db, customer_id: str, email: str, tier: str = 'starter'):
        """Create new customer"""
        customer = Customer(
            customer_id=customer_id,
            email=email,
            tier=tier
        )
        db.add(customer)
        db.commit()
        return customer
    
    @staticmethod
    def get_customer(db, customer_id: str):
        """Get customer by ID"""
        return db.query(Customer).filter(Customer.customer_id == customer_id).first()
    
    @staticmethod
    def update_customer_tier(db, customer_id: str, new_tier: str):
        """Update customer tier"""
        customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if customer:
            customer.tier = new_tier
            customer.updated_at = datetime.utcnow()
            db.commit()
        return customer
    
    @staticmethod
    def get_usage(db, customer_id: str, month: int, year: int):
        """Get usage for customer in specific month"""
        return db.query(UsageTracking).filter(
            UsageTracking.customer_id == customer_id,
            UsageTracking.month == month,
            UsageTracking.year == year
        ).first()
    
    @staticmethod
    def increment_usage(db, customer_id: str):
        """Increment usage counter"""
        now = datetime.utcnow()
        usage = DatabaseOperations.get_usage(db, customer_id, now.month, now.year)
        
        if not usage:
            usage = UsageTracking(
                customer_id=customer_id,
                month=now.month,
                year=now.year,
                reports_generated=1,
                last_report_time=now
            )
            db.add(usage)
        else:
            usage.reports_generated += 1
            usage.last_report_time = now
        
        db.commit()
        return usage
    
    @staticmethod
    def save_report(db, report_id: str, customer_id: str, portfolio_value: float, tier: str):
        """Save report to history"""
        report = Report(
            report_id=report_id,
            customer_id=customer_id,
            portfolio_value=portfolio_value,
            report_type='portfolio_analysis',
            tier_used=tier
        )
        db.add(report)
        db.commit()
        return report

if __name__ == '__main__':
    # Initialize database when run directly
    init_db()
    print("AlphaSheet Intelligence™ Database initialized!")