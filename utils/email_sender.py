"""
AlphaSheet Intelligence™ -- Email Sender Utility
Handles all email scheduling and sending for different tiers
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ADD THIS IMPORT AT THE TOP ← HERE!
from visual_branding import AlphaSheetVisualBranding

logger = logging.getLogger(__name__)

class EmailScheduler:
    """Manages email sending schedules based on tier"""
    
    def __init__(self, tier: str, email_config: Optional[Dict] = None):
        """
        Initialize email scheduler
        
        Args:
            tier: Customer subscription tier ('starter', 'growth', 'premium')
            email_config: Optional email server configuration
        """
        self.tier = tier.lower()
        self.email_config = email_config or {}
        
        # Define email schedules for each tier
        self.schedules = {
            'starter': {
                'frequency': 'weekly',
                'day': 'monday',
                'time': '09:00',
                'type': 'summary'
            },
            'growth': {
                'frequency': 'weekly',
                'day': 'monday',
                'time': '08:00',
                'type': 'intelligence_brief'
            },
            'premium': {
                'frequency': 'daily',
                'time': '07:00',
                'type': 'market_brief',
                'alerts': True
            }
        }
    
    def should_send_email(self, last_sent: Optional[datetime] = None) -> bool:
        """
        Check if email should be sent based on schedule
        
        Args:
            last_sent: When the last email was sent
            
        Returns:
            Boolean indicating if email should be sent
        """
        if not last_sent:
            return True
            
        now = datetime.now()
        schedule = self.schedules.get(self.tier, {})
        
        if schedule.get('frequency') == 'daily':
            # Send daily for premium
            return (now - last_sent).days >= 1
        elif schedule.get('frequency') == 'weekly':
            # Send weekly for starter and growth
            return (now - last_sent).days >= 7
        
        return False
    
    def get_email_content_type(self) -> str:
        """Get the type of email content for this tier"""
        schedule = self.schedules.get(self.tier, {})
        return schedule.get('type', 'summary')
    
    def format_weekly_summary(self, portfolio_data: Dict[str, Any]) -> str:
        """
        Format a simple weekly summary for Starter tier
        WITH ALPHASHEET BRANDING ← UPDATED!
        """
        # ADD BRANDING HERE ← 
        subject = AlphaSheetVisualBranding.get_email_subject('weekly_summary', 'starter')
        
        content = f"""
        Weekly Portfolio Summary - AlphaSheet Intelligence™
        ====================================================
        
        Portfolio Value: ${portfolio_data.get('total_value', 0):,.2f}
        Weekly Change: {portfolio_data.get('weekly_change', 0):.2%}
        
        Top Performer: {portfolio_data.get('top_performer', 'N/A')}
        Goal Progress: {portfolio_data.get('goal_progress', 0):.1%}
        
        Reports Used: {portfolio_data.get('reports_used', 0)} of {portfolio_data.get('reports_limit', 2)}
        
        Log in to generate your detailed report.


        --
        Powered by AlphaSheet Intelligence™
        Part of the AlphaSheet AI™ Suite
        """
        return content
    
    def format_intelligence_brief(self, portfolio_data: Dict[str, Any]) -> str:
        """
        Format an intelligence brief for Growth tier
        WITH ALPHASHEET BRANDING ← UPDATED!
        """
        # GET BRANDED HEADER ← HERE!
        header = AlphaSheetVisualBranding.get_email_header_html()
        
        # GET BRANDED FOOTER ← HERE!
        footer = AlphaSheetVisualBranding.get_footer_html()

        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            
            {header}
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>Portfolio Performance</h3>
                <p><strong>Value:</strong> ${portfolio_data.get('total_value', 0):,.2f}</p>
                <p><strong>Weekly Return:</strong> {portfolio_data.get('weekly_change', 0):.2%}</p>
            </div>
            
            <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>Key Insights</h3>
                <ul>
                    <li>{portfolio_data.get('insight_1', 'Market conditions favorable')}</li>
                    <li>{portfolio_data.get('insight_2', 'Rebalancing may be needed')}</li>
                    <li>{portfolio_data.get('insight_3', 'Tax harvesting opportunity identified')}</li>
                </ul>
            </div>
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>Recommended Actions</h3>
                <p>{portfolio_data.get('recommendation', 'Review your portfolio allocation')}</p>
            </div>
            
            {footer}

        </body>
        </html>
        """
        return html_content
    
    def format_market_brief(self, portfolio_data: Dict[str, Any], trigger: Optional[str] = None) -> str:
        """
        Format a market brief for Premium tier
        WITH ALPHASHEET BRANDING ← UPDATED!
        """
        # GET TIER-SPECIFIC HEADER ← HERE!
        header = AlphaSheetVisualBranding.get_report_header_html(
            tier='premium',
            customer_name=portfolio_data.get('customer_name')
        )
        
        # GET BRANDED FOOTER ← HERE!
        footer = AlphaSheetVisualBranding.get_footer_html()

        trigger_message = ""
        if trigger:
            trigger_message = f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ Alert Trigger:</strong> {trigger}
            </div>
            """
        
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            {header}
            
            {trigger_message}
            
            <!-- Using branded metric cards ← HERE! -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                {AlphaSheetVisualBranding.get_metric_card_html(
                    "Portfolio Value",
                    f"${portfolio_data.get('total_value', 0):,.2f}",
                    portfolio_data.get('daily_change_pct', 0) * 100,
                    'premium'
                )}
                {AlphaSheetVisualBranding.get_metric_card_html(
                    "Day Change",
                    f"{portfolio_data.get('daily_change', 0):.2%}",
                    None,
                    'premium'
                )}
                {AlphaSheetVisualBranding.get_metric_card_html(
                    "Volatility",
                    f"{portfolio_data.get('volatility', 0):.2%}",
                    None,
                    'premium'
                )}
            </div>
            
            <div style="background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>ML Signals</h3>
                <p><strong>Signal:</strong> {portfolio_data.get('ml_signal', 'HOLD')}</p>
                <p><strong>Confidence:</strong> {portfolio_data.get('ml_confidence', 0):.1%}</p>
                <p><strong>Regime:</strong> {portfolio_data.get('market_regime', 'Neutral')}</p>
            </div>
            
            <div style="background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>Risk Metrics</h3>
                <p><strong>VaR (95%):</strong> ${portfolio_data.get('var_95', 0):,.2f}</p>
                <p><strong>Max Drawdown:</strong> {portfolio_data.get('max_drawdown', 0):.2%}</p>
                <p><strong>Sharpe Ratio:</strong> {portfolio_data.get('sharpe', 0):.2f}</p>
            </div>
            
            <!-- Using branded button ← HERE! -->
            <div style="text-align: center; margin: 30px 0;">
                <a href="{portfolio_data.get('report_link', '#')}" 
                    style="{AlphaSheetVisualBranding.get_button_style('premium')}">
                    View Full Report
                </a>
            </div>

            {footer}

        </body>
        </html>
        """
        return html_content
    
    def send_email(self, to_email: str, subject: str, content: str, is_html: bool = False) -> bool:
        """
        Send an email WITH ALPHASHEET BRANDING
        """
        try:
            # GET BRANDED SUBJECT ← HERE!
            if not subject.startswith('AlphaSheet'):
                subject = f"{subject} - AlphaSheet Intelligence™"

            # This is a placeholder - in production, you'd use a real email service
            # like SendGrid, AWS SES, or SMTP
            logger.info(f"Sending email to {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Content type: {'HTML' if is_html else 'Text'}")
            
            # In production, uncomment and configure this:
            """
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('from_email', 'noreply@portfolioagent.com')
            msg['To'] = to_email
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(content, 'html'))
            else:
                msg.attach(MIMEText(content, 'plain'))
            
            # Send via SMTP
            with smtplib.SMTP(self.email_config.get('smtp_host'), self.email_config.get('smtp_port')) as server:
                server.starttls()
                server.login(self.email_config.get('username'), self.email_config.get('password'))
                server.send_message(msg)
            """
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def get_next_send_time(self) -> datetime:
        """Get the next scheduled email send time"""
        now = datetime.now()
        schedule = self.schedules.get(self.tier, {})
        
        if schedule.get('frequency') == 'daily':
            # Next day at scheduled time
            next_time = now + timedelta(days=1)
            time_parts = schedule.get('time', '07:00').split(':')
            next_time = next_time.replace(hour=int(time_parts[0]), minute=int(time_parts[1]))
        elif schedule.get('frequency') == 'weekly':
            # Next week on scheduled day
            days_ahead = 7  # Default to 7 days
            if schedule.get('day'):
                days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                target_day = days_of_week.index(schedule['day'].lower())
                days_ahead = (target_day - now.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
            next_time = now + timedelta(days=days_ahead)
            time_parts = schedule.get('time', '09:00').split(':')
            next_time = next_time.replace(hour=int(time_parts[0]), minute=int(time_parts[1]))
        else:
            next_time = now + timedelta(days=1)
        
        return next_time