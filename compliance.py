"""
Legal Compliance Wrapper for Portfolio Intelligence
Ensures all outputs are educational, not investment advice
Multi-region compliance support
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


class ComplianceWrapper:
    """
    Ensures all portfolio analysis outputs comply with financial regulations
    by framing content as educational rather than personalized advice
    """
    
    def __init__(self, region: str = 'US'):
        """
        Initialize compliance wrapper with region-specific rules
        
        Args:
            region: User's region for compliance (US, EU, UK, AU, CA, SG, HK, IN, CN, NZ, CH, JP)
        """
        self.region = region.upper()
        self.disclaimers = self._get_regional_disclaimers()
        self.replacement_phrases = self._get_replacement_phrases()
        
    def _get_regional_disclaimers(self) -> Dict[str, str]:
        """Get region-specific disclaimers"""
        base_disclaimer = "This report is for educational and informational purposes only. It does not constitute personalized investment advice."
        
        regional_additions = {
            'US': "Securities offered through registered broker-dealers. Past performance does not guarantee future results.",
            'EU': "This information does not constitute investment advice under MiFID II regulations.",
            'UK': "This is not regulated advice under FCA rules. Capital at risk.",
            'AU': "This is general information only and does not consider your personal circumstances under ASIC regulations.",
            'CA': "This does not constitute advice under Canadian securities regulations.",
            'SG': "This is not personalized advice under MAS regulations.",
            'HK': "This does not constitute advice under SFC regulations.",
            'IN': "This is not investment advice under SEBI regulations.",
            'JP': "This is not investment advice under FSA regulations.",
            'CH': "This is not investment advice under FINMA regulations.",
            'NZ': "This is not personalized financial advice under FMA regulations.",
            'CN': "This is educational content only, not investment advice under CSRC regulations."
        }
        
        regional_disclaimer = regional_additions.get(self.region, "Consult local regulations.")
        
        return {
            'header': f"{base_disclaimer} {regional_disclaimer}",
            'footer': "Always consult with a qualified financial advisor before making investment decisions.",
            'ai_insights': "These insights are based on mathematical analysis and historical patterns, not personalized recommendations.",
            'risk_analysis': "Risk metrics are educational tools. Your actual risk tolerance may differ.",
            'tax_section': "This is general tax education. Consult a tax professional for your specific situation.",
            'rebalancing': "This shows mathematical deviations from target allocations, not recommendations to trade."
        }
    
    def _get_replacement_phrases(self) -> Dict[str, str]:
        """Get phrases to replace for compliance"""
        return {
            # Direct advice to educational
            "you should": "investors often consider",
            "you must": "it's typically important to",
            "you need to": "one might consider",
            "we recommend": "financial theory suggests",
            "our recommendation": "general observation",
            
            # Action words to educational
            "buy": "research opportunities in",
            "sell": "review positions in",
            "purchase": "consider researching",
            "dispose": "evaluate holdings in",
            "acquire": "explore options for",
            
            # Certainty to probability
            "will": "may",
            "definitely": "potentially",
            "certainly": "possibly",
            "guaranteed": "historically observed",
            
            # Personal to general
            "your portfolio": "this portfolio",
            "your investments": "these investments",
            "your risk": "the portfolio risk",
            "your returns": "the returns",
            
            # Advisory to educational
            "advice": "educational insight",
            "recommendation": "observation",
            "suggest you": "investors often",
            "advise": "note that",
            "optimal for you": "mathematically optimal"
        }
    
    def wrap_text(self, text: str, context: str = 'general') -> str:
        """
        Wrap text with compliance-safe language
        
        Args:
            text: Original text to wrap
            context: Context of the text (general, risk, tax, rebalancing, etc.)
            
        Returns:
            Compliance-wrapped text
        """
        if not text:
            return text
            
        # Apply replacement phrases
        wrapped_text = text
        for original, replacement in self.replacement_phrases.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            wrapped_text = pattern.sub(replacement, wrapped_text)
        
        # Add context-specific prefix if needed
        context_prefixes = {
            'risk': "From an educational perspective: ",
            'tax': "For general tax awareness: ",
            'rebalancing': "Mathematical analysis shows: ",
            'performance': "Historical data indicates: ",
            'ai_insight': "Based on pattern analysis: "
        }
        
        if context in context_prefixes:
            wrapped_text = context_prefixes[context] + wrapped_text
            
        return wrapped_text
    
    def wrap_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap an insight dictionary with compliance language
        
        Args:
            insight: Dictionary containing insight data
            
        Returns:
            Compliance-wrapped insight
        """
        wrapped_insight = insight.copy()
        
        # Wrap text fields
        text_fields = ['description', 'rationale', 'action', 'observation', 'finding']
        for field in text_fields:
            if field in wrapped_insight and isinstance(wrapped_insight[field], str):
                wrapped_insight[field] = self.wrap_text(wrapped_insight[field], 'ai_insight')
        
        # Add educational flag
        wrapped_insight['educational_content'] = True
        wrapped_insight['personalized_advice'] = False
        
        return wrapped_insight
    
    def wrap_section(self, section_name: str, content: Any) -> Dict[str, Any]:
        """
        Wrap an entire report section with appropriate disclaimers
        
        Args:
            section_name: Name of the section
            content: Section content (can be string, dict, or list)
            
        Returns:
            Wrapped section with disclaimers
        """
        # Section-specific disclaimers
        section_disclaimers = {
            'executive_summary': "This summary provides educational insights based on portfolio analysis.",
            'risk_analysis': "These risk metrics are educational calculations based on historical data.",
            'performance': "Past performance is shown for educational purposes and does not predict future results.",
            'rebalancing': "This analysis shows mathematical deviations from target allocations.",
            'tax_optimization': "This is general tax education. Tax laws vary by jurisdiction and individual circumstances.",
            'ai_insights': "These are pattern-based observations for educational purposes.",
            'trading_signals': "These signals are based on technical analysis education, not trading recommendations.",
            'peer_comparison': "Anonymous peer data shown for educational context only."
        }
        
        wrapped_section = {
            'disclaimer': section_disclaimers.get(section_name, self.disclaimers['header']),
            'content': self._wrap_content(content, section_name),
            'educational': True
        }
        
        return wrapped_section
    
    def _wrap_content(self, content: Any, context: str) -> Any:
        """Recursively wrap content based on type"""
        if isinstance(content, str):
            return self.wrap_text(content, context)
        elif isinstance(content, dict):
            return {k: self._wrap_content(v, context) for k, v in content.items()}
        elif isinstance(content, list):
            return [self._wrap_content(item, context) for item in content]
        else:
            return content
    
    def wrap_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap an entire report with compliance
        
        Args:
            report: Complete report dictionary
            
        Returns:
            Compliance-wrapped report
        """
        wrapped_report = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'report_type': 'educational',
                'is_advice': False,
                'region': self.region,
                'disclaimer': self.disclaimers['header']
            },
            'sections': {}
        }
        
        # Wrap each section
        for section_name, section_content in report.items():
            if section_name != 'metadata':  # Don't double-wrap metadata
                wrapped_report['sections'][section_name] = self.wrap_section(
                    section_name, 
                    section_content
                )
        
        # Add footer disclaimer
        wrapped_report['footer'] = {
            'disclaimer': self.disclaimers['footer'],
            'regulatory_notice': self._get_regulatory_notice(),
            'educational_resources': self._get_educational_resources()
        }
        
        return wrapped_report
    
    def _get_regulatory_notice(self) -> str:
        """Get region-specific regulatory notice"""
        notices = {
            'US': "This service is not registered with the SEC or any state securities regulator.",
            'EU': "This service does not provide MiFID II regulated investment advice.",
            'UK': "This service is not authorized by the FCA to provide investment advice.",
            'AU': "This service does not hold an Australian Financial Services License (AFSL).",
            'CA': "This service is not registered with Canadian provincial securities commissions.",
            'SG': "This service is not licensed by the Monetary Authority of Singapore.",
            'HK': "This service is not licensed by the Securities and Futures Commission.",
            'IN': "This service is not registered with SEBI as an investment advisor.",
            'JP': "This service is not registered with the Financial Services Agency of Japan.",
            'CH': "This service is not supervised by FINMA.",
            'NZ': "This service does not provide financial advice under the Financial Markets Conduct Act.",
            'CN': "This service is not approved by the China Securities Regulatory Commission."
        }
        
        return notices.get(self.region, "This service provides educational content only.")
    
    def _get_educational_resources(self) -> List[str]:
        """Get list of educational resources by region"""
        base_resources = [
            "Consult a qualified financial advisor for personalized advice",
            "Review our educational guides on portfolio management",
            "Understand your local tax obligations"
        ]
        
        regional_resources = {
            'US': ["Visit SEC.gov for investor education", "Check FINRA.org for investment basics"],
            'UK': ["Visit FCA consumer pages", "Review MoneyHelper.org.uk"],
            'AU': ["Visit MoneySmart.gov.au", "Check ASIC investor resources"],
            'CA': ["Visit GetSmarterAboutMoney.ca", "Review CSA investor resources"],
            'EU': ["Check your national regulator's investor education portal"],
            'SG': ["Visit MoneySense.gov.sg"],
            'HK': ["Visit the Investor and Financial Education Council (IFEC)"],
            'IN': ["Visit SEBI investor education portal"],
            'default': ["Check your local financial regulator's educational resources"]
        }
        
        return base_resources + regional_resources.get(self.region, regional_resources['default'])
    
    def validate_output(self, text: str) -> Dict[str, Any]:
        """
        Validate that output doesn't contain prohibited terms
        
        Args:
            text: Text to validate
            
        Returns:
            Validation result with any issues found
        """
        prohibited_terms = [
            'guarantee', 'guaranteed return', 'risk-free', 'sure thing',
            'can\'t lose', 'definitely will', 'hot tip', 'insider',
            'once in a lifetime', 'act now', 'don\'t miss out'
        ]
        
        issues_found = []
        for term in prohibited_terms:
            if term.lower() in text.lower():
                issues_found.append(f"Found prohibited term: '{term}'")
        
        return {
            'valid': len(issues_found) == 0,
            'issues': issues_found,
            'cleaned_text': self.wrap_text(text) if issues_found else text
        }
    
    def get_terms_of_service_snippet(self) -> str:
        """Get key terms of service points for the region"""
        return f"""
By using this service, you acknowledge that:
1. This is an educational platform providing general information
2. Nothing herein constitutes personalized investment advice
3. {self._get_regulatory_notice()}
4. You should consult qualified professionals before making investment decisions
5. Past performance does not indicate future results
6. All investments carry risk of loss
        """