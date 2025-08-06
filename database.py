"""
AlphaSheet Intelligence™ - Database Models
PostgreSQL database setup with Flask-SQLAlchemy for production
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship

# Initialize Flask-SQLAlchemy
db = SQLAlchemy()

def init_db(app, database_url=None):
    """Initialize database with Flask app"""
    if database_url is None:
        database_url = os.getenv('DATABASE_URL', 'sqlite:///alphasheet.db')
    
    # Fix for SQLAlchemy (Railway provides postgres://, SQLAlchemy needs postgresql://)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    return db

# Database Models
class User(db.Model):
    """User model (matches app.py expectations)"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    tier = Column(String(50), default='starter')
    region = Column(String(50), default='US')
    stripe_customer_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship('Subscription', backref='user', lazy=True)
    usage_tracking = relationship('UsageTracking', backref='user', lazy=True)
    reports = relationship('Report', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'

class Subscription(db.Model):
    """Subscription model for Stripe integration"""
    __tablename__ = 'subscriptions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stripe_subscription_id = Column(String(255), unique=True, nullable=True)
    tier = Column(String(50), nullable=False)
    status = Column(String(50), default='active')  # active, cancelled, past_due
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Subscription {self.tier} for User {self.user_id}>'

class UsageTracking(db.Model):
    """Usage tracking model"""
    __tablename__ = 'usage_tracking'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    customer_id = Column(String(100), nullable=False)  # Keep for backwards compatibility
    endpoint = Column(String(100), nullable=True)
    tier = Column(String(50), nullable=False)
    month = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    reports_generated = Column(Integer, default=0)
    last_report_time = Column(DateTime, nullable=True)
    portfolio_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UsageTracking User {self.user_id} - {self.month}/{self.year}>'

class Report(db.Model):
    """Report history model"""
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    customer_id = Column(String(100), nullable=False)  # Keep for backwards compatibility
    generated_at = Column(DateTime, default=datetime.utcnow)
    portfolio_value = Column(Float, nullable=True)
    report_type = Column(String(50), default='portfolio_analysis')
    tier = Column(String(50), nullable=False)
    
    # Optional: Store report content
    html_content = Column(Text, nullable=True)
    json_content = Column(Text, nullable=True)
    
    def __repr__(self):
        return f'<Report {self.report_id}>'

# Keep legacy Customer model for compatibility (maps to User)
class Customer(db.Model):
    """Legacy Customer model - redirects to User table"""
    __tablename__ = 'customers_legacy'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    tier = Column(String(50), default='starter')
    region = Column(String(50), default='US')
    stripe_customer_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database operations helper class
class DatabaseOperations:
    """Database helper functions"""
    
    @staticmethod
    def create_user(email: str, customer_id: str = None, tier: str = 'starter'):
        """Create new user"""
        if not customer_id:
            # Generate customer_id from email if not provided
            import hashlib
            customer_id = hashlib.md5(email.encode()).hexdigest()[:12]
        
        user = User(
            customer_id=customer_id,
            email=email,
            tier=tier
        )
        db.session.add(user)
        db.session.commit()
        return user
    
    @staticmethod
    def get_user_by_email(email: str):
        """Get user by email"""
        return User.query.filter_by(email=email).first()
    
    @staticmethod
    def get_user_by_customer_id(customer_id: str):
        """Get user by customer ID"""
        return User.query.filter_by(customer_id=customer_id).first()
    
    @staticmethod
    def update_user_tier(user_id: int, new_tier: str):
        """Update user tier"""
        user = User.query.get(user_id)
        if user:
            user.tier = new_tier
            user.updated_at = datetime.utcnow()
            db.session.commit()
        return user
    
    @staticmethod
    def get_usage(user_id: int, month: int = None, year: int = None):
        """Get usage for user in specific month"""
        if month is None or year is None:
            now = datetime.utcnow()
            month = now.month
            year = now.year
        
        return UsageTracking.query.filter_by(
            user_id=user_id,
            month=month,
            year=year
        ).first()
    
    @staticmethod
    def increment_usage(user_id: int, customer_id: str, tier: str):
        """Increment usage counter"""
        now = datetime.utcnow()
        usage = DatabaseOperations.get_usage(user_id, now.month, now.year)
        
        if not usage:
            usage = UsageTracking(
                user_id=user_id,
                customer_id=customer_id,
                tier=tier,
                month=now.month,
                year=now.year,
                reports_generated=1,
                last_report_time=now
            )
            db.session.add(usage)
        else:
            usage.reports_generated += 1
            usage.last_report_time = now
        
        db.session.commit()
        return usage
    
    @staticmethod
    def save_report(user_id: int, customer_id: str, report_id: str, 
                   portfolio_value: float, tier: str, html_content: str = None):
        """Save report to history"""
        import uuid
        if not report_id:
            report_id = str(uuid.uuid4())[:12]
        
        report = Report(
            report_id=report_id,
            user_id=user_id,
            customer_id=customer_id,
            portfolio_value=portfolio_value,
            report_type='portfolio_analysis',
            tier=tier,
            html_content=html_content
        )
        db.session.add(report)
        db.session.commit()
        return report
    
    @staticmethod
    def create_subscription(user_id: int, tier: str, stripe_subscription_id: str = None):
        """Create subscription for user"""
        subscription = Subscription(
            user_id=user_id,
            tier=tier,
            stripe_subscription_id=stripe_subscription_id,
            status='active'
        )
        db.session.add(subscription)
        db.session.commit()
        return subscription
    
    @staticmethod
    def get_active_subscription(user_id: int):
        """Get active subscription for user"""
        return Subscription.query.filter_by(
            user_id=user_id,
            status='active'
        ).first()
    
    @staticmethod
    def cancel_subscription(subscription_id: int):
        """Cancel subscription"""
        subscription = Subscription.query.get(subscription_id)
        if subscription:
            subscription.status = 'cancelled'
            subscription.updated_at = datetime.utcnow()
            db.session.commit()
        return subscription
    
    @staticmethod
    def get_usage_summary(user_id: int):
        """Get usage summary for current month"""
        now = datetime.utcnow()
        usage = DatabaseOperations.get_usage(user_id, now.month, now.year)
        
        if usage:
            return {
                'reports_generated': usage.reports_generated,
                'last_report': usage.last_report_time.isoformat() if usage.last_report_time else None,
                'month': usage.month,
                'year': usage.year
            }
        return {
            'reports_generated': 0,
            'last_report': None,
            'month': now.month,
            'year': now.year
        }
    
    @staticmethod
    def get_user_reports(user_id: int, limit: int = 10):
        """Get recent reports for user"""
        return Report.query.filter_by(user_id=user_id)\
                          .order_by(Report.generated_at.desc())\
                          .limit(limit)\
                          .all()

# Migration helper for existing data
def migrate_from_legacy():
    """Migrate from old Customer table to new User table if needed"""
    try:
        # Check if we have any customers in legacy table
        legacy_customers = Customer.query.all()
        
        for customer in legacy_customers:
            # Check if user already exists
            existing_user = User.query.filter_by(email=customer.email).first()
            if not existing_user:
                # Create new user from customer data
                new_user = User(
                    customer_id=customer.customer_id,
                    email=customer.email,
                    tier=customer.tier,
                    region=customer.region,
                    stripe_customer_id=customer.stripe_customer_id,
                    created_at=customer.created_at,
                    updated_at=customer.updated_at
                )
                db.session.add(new_user)
        
        db.session.commit()
        print(f"Migrated {len(legacy_customers)} customers to users table")
        
    except Exception as e:
        print(f"Migration skipped or failed: {e}")
        db.session.rollback()

if __name__ == '__main__':
    # This won't work standalone anymore - needs Flask app context
    print("Database module loaded. Use with Flask app for initialization.")