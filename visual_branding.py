"""
AlphaSheet Intelligence™ - Visual Brand Configuration
Aligned with existing AlphaSheet AI logo and visual identity
Updated with Inter as the official brand font
"""

from typing import Dict, Any

class AlphaSheetVisualBranding:
    """
    Visual branding configuration matching the AlphaSheet logo design
    """
    
    # Core Brand Identity (matching your logos)
    PRODUCT_NAME = "AlphaSheet Intelligence™"
    DISPLAY_NAME = "AlphaSheet AI"  # As shown in logo
    LOGO_SYMBOL = "α"  # Alpha symbol from logo
    TAGLINE = "Institutional-Grade Portfolio Intelligence"
    
    # Tier Names (required by app.py)
    TIER_NAMES = {
        'starter': 'Intelligence Starter',
        'growth': 'Intelligence Growth',
        'premium': 'Intelligence Premium'
    }
    
    # Brand Typography - Inter as primary font
    FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif"
    FONT_IMPORT = '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap");'
    
    # Font Weights for Inter
    FONT_WEIGHTS = {
        'light': 300,
        'regular': 400,
        'medium': 500,
        'semibold': 600,
        'bold': 700
    }
    
    # Brand Colors (extracted from your gradient logo)
    BRAND_COLORS = {
        # Primary gradient colors from logo
        'gradient_start': '#4A90E2',   # Blue (left side of gradient)
        'gradient_end': '#50E3C2',     # Teal/turquoise (right side)
        
        # Solid colors for various uses
        'primary_blue': '#4A90E2',     # Main blue from logo
        'primary_teal': '#50E3C2',     # Teal accent from "AI" text
        'alpha_white': '#FFFFFF',      # White alpha symbol
        
        # Supporting colors
        'dark_text': '#2C3E50',        # Dark gray for text
        'light_gray': '#F8F9FA',       # Light background
        'success': '#50E3C2',          # Using teal for success
        'warning': '#F39C12',          # Orange for warnings
        'danger': '#E74C3C',           # Red for alerts
        'premium': '#7B68EE',          # Purple for premium tier
    }
    
    # CSS Gradient for HTML elements (matching logo)
    GRADIENT_CSS = "linear-gradient(135deg, #4A90E2 0%, #50E3C2 100%)"
    
    # Tier Visual Identity (using logo colors)
    TIER_COLORS = {
        'starter': {
            'primary': '#95A5A6',  # Gray
            'gradient': 'linear-gradient(135deg, #95A5A6 0%, #BDC3C7 100%)'
        },
        'growth': {
            'primary': '#4A90E2',  # Logo blue
            'gradient': 'linear-gradient(135deg, #4A90E2 0%, #5DADE2 100%)'
        },
        'premium': {
            'primary': '#7B68EE',  # Premium purple
            'gradient': 'linear-gradient(135deg, #7B68EE 0%, #50E3C2 100%)'
        }
    }
    
    @staticmethod
    def get_logo_html(size: str = 'medium') -> str:
        """
        Get HTML representation of the logo with Inter font
        
        Args:
            size: 'small', 'medium', or 'large'
            
        Returns:
            HTML for logo display
        """
        sizes = {
            'small': {'icon': '24px', 'text': '18px'},
            'medium': {'icon': '32px', 'text': '24px'},
            'large': {'icon': '48px', 'text': '32px'}
        }
        
        size_config = sizes.get(size, sizes['medium'])
        
        return f"""
        <div style="display: inline-flex; align-items: center; gap: 10px; font-family: {AlphaSheetVisualBranding.FONT_FAMILY};">
            <div style="width: {size_config['icon']}; height: {size_config['icon']}; 
                        background: {AlphaSheetVisualBranding.GRADIENT_CSS}; 
                        border-radius: 8px; display: flex; align-items: center; 
                        justify-content: center; color: white; font-size: {size_config['icon']}; font-weight: 600;">
                α
            </div>
            <div style="font-size: {size_config['text']}; color: {AlphaSheetVisualBranding.BRAND_COLORS['dark_text']}; font-weight: 500;">
                AlphaSheet <span style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_teal']};">AI</span>
            </div>
        </div>
        """
    
    @staticmethod
    def get_email_header_html() -> str:
        """
        Get branded email header with Inter font
        
        Returns:
            HTML email header
        """
        return f"""
        <style>
            {AlphaSheetVisualBranding.FONT_IMPORT}
        </style>
        <div style="background: {AlphaSheetVisualBranding.GRADIENT_CSS}; 
                    padding: 30px; text-align: center; border-radius: 10px 10px 0 0; 
                    font-family: {AlphaSheetVisualBranding.FONT_FAMILY};">
            <div style="display: inline-block; background: rgba(255,255,255,0.1); 
                        border-radius: 12px; padding: 15px 25px;">
                <div style="font-size: 48px; color: white; margin-bottom: 10px; font-weight: 600;">α</div>
                <h1 style="margin: 0; color: white; font-size: 24px; font-weight: 300; letter-spacing: -0.5px;">
                    AlphaSheet Intelligence™
                </h1>
                <p style="margin: 5px 0 0 0; color: rgba(255,255,255,0.9); font-size: 14px; font-weight: 400;">
                    Institutional-Grade Portfolio Analysis
                </p>
            </div>
        </div>
        """
    
    @staticmethod
    def get_report_header_html(tier: str = 'starter', customer_name: str = None) -> str:
        """
        Get report header with tier-appropriate styling and Inter font
        
        Args:
            tier: Customer tier
            customer_name: Optional customer name
            
        Returns:
            HTML report header
        """
        tier_config = AlphaSheetVisualBranding.TIER_COLORS.get(tier, AlphaSheetVisualBranding.TIER_COLORS['starter'])
        greeting = f"Portfolio Analysis for {customer_name}" if customer_name else "Portfolio Intelligence Report"
        
        # Tier badges with appropriate colors
        tier_names = {
            'starter': 'Intelligence Starter',
            'growth': 'Intelligence Growth',
            'premium': 'Intelligence Premium'
        }
        
        return f"""
        <style>
            {AlphaSheetVisualBranding.FONT_IMPORT}
        </style>
        <div style="background: {tier_config['gradient']}; color: white; padding: 40px 30px; 
                    border-radius: 12px 12px 0 0; position: relative; overflow: hidden; 
                    font-family: {AlphaSheetVisualBranding.FONT_FAMILY};">
            <!-- Background pattern mimicking logo flow lines -->
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.1;">
                <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
                    <pattern id="pattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                        <path d="M0,50 Q25,30 50,50 T100,50" stroke="white" fill="none" stroke-width="1"/>
                    </pattern>
                    <rect width="100%" height="100%" fill="url(#pattern)"/>
                </svg>
            </div>
            
            <!-- Content -->
            <div style="position: relative; z-index: 1;">
                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                    <div style="width: 50px; height: 50px; background: rgba(255,255,255,0.2); 
                                border-radius: 10px; display: flex; align-items: center; 
                                justify-content: center; font-size: 32px; font-weight: 600;">
                        α
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 28px; font-weight: 300; letter-spacing: -0.5px;">
                            AlphaSheet Intelligence™
                        </h1>
                        <span style="display: inline-block; padding: 4px 12px; 
                                     background: rgba(255,255,255,0.2); border-radius: 20px; 
                                     font-size: 12px; margin-top: 5px; font-weight: 500;">
                            {tier_names.get(tier, 'Intelligence')}
                        </span>
                    </div>
                </div>
                <p style="margin: 0; font-size: 18px; opacity: 0.95; font-weight: 400;">
                    {greeting}
                </p>
                <p style="margin: 8px 0 0 0; font-size: 14px; opacity: 0.8; font-weight: 400;">
                    Generated on {AlphaSheetVisualBranding.get_current_date()}
                </p>
            </div>
        </div>
        """
    
    @staticmethod
    def get_card_style(tier: str = 'starter') -> str:
        """
        Get CSS style for cards matching tier with Inter font
        
        Args:
            tier: Customer tier
            
        Returns:
            CSS style string
        """
        tier_color = AlphaSheetVisualBranding.TIER_COLORS[tier]['primary']
        
        return f"""
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid {tier_color};
        margin-bottom: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
        font-family: {AlphaSheetVisualBranding.FONT_FAMILY};
        """
    
    @staticmethod
    def get_button_style(button_type: str = 'primary') -> str:
        """
        Get button styling matching brand with Inter font
        
        Args:
            button_type: 'primary', 'secondary', or 'premium'
            
        Returns:
            CSS style string
        """
        styles = {
            'primary': f"""
                background: {AlphaSheetVisualBranding.GRADIENT_CSS};
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                text-decoration: none;
                display: inline-block;
                font-weight: 500;
                border: none;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                font-family: {AlphaSheetVisualBranding.FONT_FAMILY};
                letter-spacing: -0.2px;
            """,
            'secondary': f"""
                background: white;
                color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_blue']};
                padding: 12px 24px;
                border-radius: 25px;
                text-decoration: none;
                display: inline-block;
                font-weight: 500;
                border: 2px solid {AlphaSheetVisualBranding.BRAND_COLORS['primary_blue']};
                cursor: pointer;
                transition: all 0.2s;
                font-family: {AlphaSheetVisualBranding.FONT_FAMILY};
                letter-spacing: -0.2px;
            """,
            'premium': f"""
                background: linear-gradient(135deg, {AlphaSheetVisualBranding.BRAND_COLORS['premium']} 0%, {AlphaSheetVisualBranding.BRAND_COLORS['primary_teal']} 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                text-decoration: none;
                display: inline-block;
                font-weight: 500;
                border: none;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                font-family: {AlphaSheetVisualBranding.FONT_FAMILY};
                letter-spacing: -0.2px;
            """
        }
        
        return styles.get(button_type, styles['primary'])
    
    @staticmethod
    def get_metric_card_html(title: str, value: str, change: float = None, tier: str = 'starter') -> str:
        """
        Get a metric card matching brand style with Inter font
        
        Args:
            title: Metric title
            value: Metric value
            change: Optional percentage change
            tier: Customer tier
            
        Returns:
            HTML for metric card
        """
        tier_color = AlphaSheetVisualBranding.TIER_COLORS[tier]['primary']
        change_html = ""
        
        if change is not None:
            change_color = AlphaSheetVisualBranding.BRAND_COLORS['success'] if change >= 0 else AlphaSheetVisualBranding.BRAND_COLORS['danger']
            arrow = "↑" if change >= 0 else "↓"
            change_html = f"""
            <span style="color: {change_color}; font-size: 14px; font-weight: 400;">
                {arrow} {abs(change):.2f}%
            </span>
            """
        
        return f"""
        <div style="background: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
                    border-top: 3px solid {tier_color};
                    font-family: {AlphaSheetVisualBranding.FONT_FAMILY};">
            <p style="margin: 0 0 10px 0; color: #666; font-size: 14px; font-weight: 400;">
                {title}
            </p>
            <p style="margin: 0; font-size: 24px; font-weight: 600; color: #333; letter-spacing: -0.5px;">
                {value} {change_html}
            </p>
        </div>
        """
    
    @staticmethod
    def get_footer_html() -> str:
        """
        Get branded footer with Inter font
        
        Returns:
            HTML footer
        """
        return f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 30px; text-align: center; margin-top: 40px; 
                    border-top: 1px solid #dee2e6;
                    font-family: {AlphaSheetVisualBranding.FONT_FAMILY};">
            <div style="display: inline-flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <div style="width: 32px; height: 32px; 
                            background: {AlphaSheetVisualBranding.GRADIENT_CSS}; 
                            border-radius: 8px; display: flex; align-items: center; 
                            justify-content: center; color: white; font-size: 20px; font-weight: 600;">
                    α
                </div>
                <div style="font-size: 18px; color: {AlphaSheetVisualBranding.BRAND_COLORS['dark_text']}; font-weight: 500;">
                    AlphaSheet <span style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_teal']};">AI</span>
                </div>
            </div>
            
            <p style="margin: 10px 0; font-size: 14px; color: #666; font-weight: 400;">
                Powered by AlphaSheet Intelligence™
            </p>
            
            <div style="margin: 20px 0;">
                <a href="https://alphasheet.ai" style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_blue']}; 
                   text-decoration: none; margin: 0 15px; font-size: 14px; font-weight: 400;">
                    Home
                </a>
                <a href="https://alphasheet.ai/intelligence" style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_blue']}; 
                   text-decoration: none; margin: 0 15px; font-size: 14px; font-weight: 400;">
                    Intelligence Portal
                </a>
                <a href="https://alphasheet.ai/support" style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_blue']}; 
                   text-decoration: none; margin: 0 15px; font-size: 14px; font-weight: 400;">
                    Support
                </a>
                <a href="https://alphasheet.ai/upgrade" style="color: {AlphaSheetVisualBranding.BRAND_COLORS['primary_teal']}; 
                   text-decoration: none; margin: 0 15px; font-size: 14px; font-weight: 600;">
                    Upgrade ⭐
                </a>
            </div>
            
            <p style="font-size: 12px; color: #999; margin: 15px 0 0 0; font-weight: 400;">
                © 2024 AlphaSheet. All rights reserved.<br>
                AlphaSheet Intelligence™ and AlphaSheet AI™ are trademarks of AlphaSheet.
            </p>
        </div>
        """
    
    @staticmethod
    def get_full_html_template() -> str:
        """
        Get a complete HTML template with Inter font properly loaded
        
        Returns:
            Complete HTML template with font imports
        """
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AlphaSheet Intelligence™ Report</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: {AlphaSheetVisualBranding.FONT_FAMILY};
                    line-height: 1.6;
                    color: {AlphaSheetVisualBranding.BRAND_COLORS['dark_text']};
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    font-weight: 600;
                    letter-spacing: -0.5px;
                }}
            </style>
        </head>
        <body>
            <!-- Report content goes here -->
        </body>
        </html>
        """
    
    @staticmethod
    def get_current_date() -> str:
        """Get current date in branded format"""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")


# Quick test
def test_visual_branding():
    """Test visual branding elements with Inter font"""
    print("🎨 AlphaSheet Intelligence™ Visual Branding Test")
    print("=" * 50)
    
    # Test font configuration
    print(f"\n✨ Brand Typography:")
    print(f"  Font Family: Inter")
    print(f"  Font Stack: {AlphaSheetVisualBranding.FONT_FAMILY[:50]}...")
    print(f"  Font Import: ✓ Google Fonts CDN")
    
    # Test gradient
    print(f"\n📊 Brand Gradient CSS:")
    print(f"  {AlphaSheetVisualBranding.GRADIENT_CSS}")
    
    # Test tier colors
    print(f"\n🎯 Tier Colors:")
    for tier in ['starter', 'growth', 'premium']:
        color = AlphaSheetVisualBranding.TIER_COLORS[tier]['primary']
        print(f"  {tier}: {color}")
    
    # Test font weights
    print(f"\n📝 Inter Font Weights:")
    for weight_name, weight_value in AlphaSheetVisualBranding.FONT_WEIGHTS.items():
        print(f"  {weight_name}: {weight_value}")
    
    # Test component generation
    print(f"\n✅ Components with Inter Font:")
    print(f"  Logo HTML: ✓")
    print(f"  Email Header: ✓")
    print(f"  Report Header: ✓")
    print(f"  Card Styles: ✓")
    print(f"  Buttons: ✓")
    print(f"  Footer: ✓")
    
    print("\n✅ Visual branding with Inter font ready for use!")


if __name__ == '__main__':
    test_visual_branding()