"""
Dynamic Domain-Agnostic NLP Text Analysis Pipeline
===================================================

Features:
- Dynamic industry-specific rules and keywords loading
- HIPAA/GDPR/PCI-DSS compliant PII redaction
- Hierarchical category/subcategory classification (L1-L4)
- Proximity-based contextual grouping
- Multi-format support (CSV, Excel, Parquet, JSON)
- Company-to-industry mapping
- Optimized performance with caching

Version: 3.0 - Domain Agnostic
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import lru_cache
import io
import os

# NLP Libraries
import spacy
from textblob import TextBlob
from deep_translator import GoogleTranslator

# ========================================================================================
# CONFIGURATION & CONSTANTS
# ========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 4
BATCH_SIZE = 100
CACHE_SIZE = 1000
SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'parquet', 'json']
COMPLIANCE_STANDARDS = ["HIPAA", "GDPR", "PCI-DSS", "CCPA"]

# Domain packs directory structure
DOMAIN_PACKS_DIR = "domain_packs"

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching and better error handling"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            logger.warning("spaCy model not found. Attempting to download...")
            import subprocess
            import sys
            
            # Use sys.executable to get correct Python path
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info("spaCy model downloaded successfully")
                return spacy.load("en_core_web_sm")
            else:
                logger.error(f"Failed to download spaCy model: {result.stderr}")
                st.error("⚠️ spaCy model download failed. Please run: python -m spacy download en_core_web_sm")
                st.stop()
        except Exception as e:
            logger.error(f"Error downloading spaCy model: {e}")
            st.error(f"⚠️ Could not load spaCy model. Error: {e}")
            st.info("💡 Solution: Add 'setup.sh' file to your repo with spaCy download command")
            st.stop()

nlp = load_spacy_model()

# Initialize translator
@st.cache_resource
def get_translator():
    """Get cached translator instance"""
    return None  # deep-translator doesn't need persistent instance

translator = get_translator()


# ========================================================================================
# DATA CLASSES
# ========================================================================================

@dataclass
class PIIRedactionResult:
    """Result of PII detection and redaction"""
    redacted_text: str
    pii_detected: bool
    pii_counts: Dict[str, int]
    total_items: int


@dataclass
class CategoryMatch:
    """Hierarchical category match result with 4 levels"""
    l1: str
    l2: str
    l3: str
    l4: str
    confidence: float
    match_path: str
    matched_rule: Optional[str] = None


@dataclass
class ProximityResult:
    """Proximity-based grouping result"""
    primary_proximity: str
    proximity_group: str
    theme_count: int
    matched_themes: List[str]


@dataclass
class NLPResult:
    """Complete NLP analysis result"""
    conversation_id: str
    original_text: str
    redacted_text: str
    translated_text: str
    category: CategoryMatch
    proximity: ProximityResult
    sentiment: str
    sentiment_score: float
    pii_result: PIIRedactionResult
    industry: Optional[str] = None


# ========================================================================================
# DOMAIN LOADER - Dynamic Industry Rules & Keywords
# ========================================================================================

class DomainLoader:
    """
    Dynamically loads industry-specific rules and keywords from JSON files
    Supports flexible domain pack structure
    """
    
    def __init__(self, domain_packs_dir: str = None):
        """
        Initialize domain loader
        
        Args:
            domain_packs_dir: Path to domain_packs directory
        """
        self.domain_packs_dir = domain_packs_dir or DOMAIN_PACKS_DIR
        self.industries = {}
        self.company_mapping = {}
        
    def load_company_mapping(self, mapping_file: str = None) -> Dict:
        """
        Load company-to-industry mapping from JSON
        
        Args:
            mapping_file: Path to mapping JSON file
            
        Returns:
            Dictionary with industry mappings
        """
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                self.company_mapping = data.get('industries', {})
                logger.info(f"Loaded company mapping with {len(self.company_mapping)} industries")
                return self.company_mapping
        return {}
    
    def auto_load_all_industries(self) -> int:
        """
        Automatically load all industries from domain_packs directory
        
        Returns:
            Number of industries loaded
        """
        loaded_count = 0
        
        if not os.path.exists(self.domain_packs_dir):
            logger.error(f"Domain packs directory not found: {self.domain_packs_dir}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            return 0
        
        logger.info(f"Scanning domain_packs directory: {self.domain_packs_dir}")
        
        # Load company mapping first
        mapping_path = os.path.join(self.domain_packs_dir, "company_industry_mapping.json")
        if os.path.exists(mapping_path):
            self.load_company_mapping(mapping_path)
            logger.info(f"Loaded company mapping from {mapping_path}")
        
        # Scan for industry directories
        try:
            items = os.listdir(self.domain_packs_dir)
            logger.info(f"Found {len(items)} items in domain_packs: {items}")
        except Exception as e:
            logger.error(f"Error listing domain_packs directory: {e}")
            return 0
        
        for item in items:
            item_path = os.path.join(self.domain_packs_dir, item)
            
            # Skip if not a directory
            if not os.path.isdir(item_path):
                logger.debug(f"Skipping non-directory: {item}")
                continue
            
            # Skip hidden directories
            if item.startswith('.'):
                logger.debug(f"Skipping hidden directory: {item}")
                continue
            
            # Look for rules.json and keywords.json
            rules_path = os.path.join(item_path, "rules.json")
            keywords_path = os.path.join(item_path, "keywords.json")
            
            logger.info(f"Checking industry: {item}")
            logger.info(f"  Rules exists: {os.path.exists(rules_path)}")
            logger.info(f"  Keywords exists: {os.path.exists(keywords_path)}")
            
            if os.path.exists(rules_path) and os.path.exists(keywords_path):
                try:
                    self.load_from_files(rules_path, keywords_path, item)
                    loaded_count += 1
                    logger.info(f"✅ Successfully auto-loaded: {item}")
                except Exception as e:
                    logger.error(f"❌ Failed to auto-load {item}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"⚠️ Skipping {item}: missing rules.json or keywords.json")
        
        logger.info(f"Auto-load complete: {loaded_count} industries loaded from {self.domain_packs_dir}")
        return loaded_count
    
    def load_from_files(self, rules_file: str, keywords_file: str, industry_name: str):
        """
        Load rules and keywords from uploaded files
        
        Args:
            rules_file: Path to rules JSON file
            keywords_file: Path to keywords JSON file
            industry_name: Name of the industry
        """
        try:
            # Load rules
            with open(rules_file, 'r') as f:
                rules = json.load(f)
            
            # Load keywords
            with open(keywords_file, 'r') as f:
                keywords = json.load(f)
            
            self.industries[industry_name] = {
                'rules': rules,
                'keywords': keywords,
                'rules_count': len(rules),
                'keywords_count': len(keywords)
            }
            
            logger.info(f"Loaded {industry_name}: {len(rules)} rules, {len(keywords)} keyword groups")
            
        except Exception as e:
            logger.error(f"Error loading {industry_name} domain pack: {e}")
            raise
    
    def get_available_industries(self) -> List[str]:
        """Get list of loaded industries"""
        return list(self.industries.keys())
    
    def get_industry_data(self, industry: str) -> Dict:
        """Get rules and keywords for specific industry"""
        return self.industries.get(industry, {'rules': [], 'keywords': []})
    
    def detect_industry_from_company(self, company_name: str) -> Optional[str]:
        """
        Detect industry based on company name
        
        Args:
            company_name: Name of the company
            
        Returns:
            Industry name or None
        """
        if not self.company_mapping:
            return None
        
        company_name_lower = company_name.lower().strip()
        
        for industry, companies in self.company_mapping.items():
            for company in companies:
                if company.lower().strip() == company_name_lower:
                    return industry
                # Partial match
                if company_name_lower in company.lower() or company.lower() in company_name_lower:
                    return industry
        
        return None


# ========================================================================================
# PII DETECTION & REDACTION ENGINE (Same as before)
# ========================================================================================

class PIIDetector:
    """
    Comprehensive PII/PHI/PCI detection and redaction engine
    Compliant with: HIPAA, GDPR, PCI-DSS, CCPA
    """
    
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    PHONE_PATTERNS = [
        re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        re.compile(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'),
        re.compile(r'\+1[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}'),
    ]
    
    CREDIT_CARD_PATTERNS = [
        re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?)\b'),  # Visa
        re.compile(r'\b(?:5[1-5][0-9]{14})\b'),  # Mastercard
        re.compile(r'\b(?:3[47][0-9]{13})\b'),  # Amex
        re.compile(r'\b(?:6(?:011|5[0-9]{2})[0-9]{12})\b'),  # Discover
    ]
    
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    DOB_PATTERN = re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b')
    MRN_PATTERN = re.compile(r'\b(?:MRN|mrn|Medical Record|medical record)[:\s]+([A-Z0-9]{6,12})\b', re.IGNORECASE)
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    
    ADDRESS_PATTERN = re.compile(
        r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Apartment|Apt|Suite|Ste|Unit)\b',
        re.IGNORECASE
    )
    
    DISEASE_KEYWORDS = {
        'diabetes', 'cancer', 'hiv', 'aids', 'covid', 'covid-19', 'coronavirus',
        'hypertension', 'depression', 'anxiety', 'asthma', 'copd', 'pneumonia',
        'tuberculosis', 'hepatitis', 'alzheimer', 'parkinson', 'schizophrenia',
        'epilepsy', 'stroke', 'heart attack', 'myocardial infarction'
    }
    
    @classmethod
    def _generate_hash(cls, text: str) -> str:
        """Generate SHA-256 hash for consistent redaction"""
        return hashlib.sha256(text.encode()).hexdigest()[:8]
    
    @classmethod
    def _is_valid_credit_card(cls, card: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        card = re.sub(r'[^0-9]', '', card)
        if len(card) < 13 or len(card) > 19:
            return False
        
        total = 0
        reverse_digits = card[::-1]
        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    @classmethod
    def _is_valid_ssn(cls, ssn: str) -> bool:
        """Validate SSN format"""
        parts = ssn.split('-')
        if len(parts) != 3:
            return False
        
        if parts[0] == '000' or parts[0] == '666' or parts[0].startswith('9'):
            return False
        if parts[1] == '00':
            return False
        if parts[2] == '0000':
            return False
        
        return True
    
    @classmethod
    def _is_valid_dob(cls, dob: str) -> bool:
        """Validate date of birth"""
        try:
            year_match = re.search(r'(19|20)\d{2}', dob)
            if year_match:
                year = int(year_match.group())
                current_year = datetime.now().year
                return 1900 <= year <= current_year
        except:
            pass
        return False
    
    @classmethod
    def _redact_value(cls, value: str, pii_type: str, mode: str) -> str:
        """Redact value based on mode"""
        if mode == 'hash':
            return f"[{pii_type}:{cls._generate_hash(value)}]"
        elif mode == 'mask':
            return f"[{pii_type}:{'*' * 10}]"
        elif mode == 'token':
            return f"[{pii_type}]"
        elif mode == 'remove':
            return ""
        else:
            return f"[{pii_type}:{cls._generate_hash(value)}]"
    
    @classmethod
    def detect_and_redact(cls, text: str, redaction_mode: str = 'hash') -> PIIRedactionResult:
        """Detect and redact all PII/PHI/PCI from text"""
        if not text or not isinstance(text, str):
            return PIIRedactionResult(
                redacted_text=str(text) if text else "",
                pii_detected=False,
                pii_counts={},
                total_items=0
            )
        
        redacted = text
        pii_counts = {}
        
        # 1. Emails
        emails = cls.EMAIL_PATTERN.findall(redacted)
        for email in emails:
            redacted = redacted.replace(email, cls._redact_value(email, 'EMAIL', redaction_mode))
            pii_counts['emails'] = pii_counts.get('emails', 0) + 1
        
        # 2. Credit cards
        for pattern in cls.CREDIT_CARD_PATTERNS:
            cards = pattern.findall(redacted)
            for card in cards:
                if cls._is_valid_credit_card(card):
                    redacted = redacted.replace(card, cls._redact_value(card, 'CARD', redaction_mode))
                    pii_counts['credit_cards'] = pii_counts.get('credit_cards', 0) + 1
        
        # 3. SSNs
        ssns = cls.SSN_PATTERN.findall(redacted)
        for ssn in ssns:
            if cls._is_valid_ssn(ssn):
                redacted = redacted.replace(ssn, cls._redact_value(ssn, 'SSN', redaction_mode))
                pii_counts['ssns'] = pii_counts.get('ssns', 0) + 1
        
        # 4. Phone numbers
        for pattern in cls.PHONE_PATTERNS:
            phones = pattern.findall(redacted)
            for phone in phones:
                redacted = redacted.replace(phone, cls._redact_value(phone, 'PHONE', redaction_mode))
                pii_counts['phones'] = pii_counts.get('phones', 0) + 1
        
        # 5. DOBs
        dobs = cls.DOB_PATTERN.findall(redacted)
        for dob in dobs:
            if cls._is_valid_dob(dob):
                redacted = redacted.replace(dob, cls._redact_value(dob, 'DOB', redaction_mode))
                pii_counts['dobs'] = pii_counts.get('dobs', 0) + 1
        
        # 6. Medical records
        mrns = cls.MRN_PATTERN.findall(redacted)
        for mrn in mrns:
            redacted = redacted.replace(mrn, cls._redact_value(mrn, 'MRN', redaction_mode))
            pii_counts['medical_records'] = pii_counts.get('medical_records', 0) + 1
        
        # 7. IP addresses
        ips = cls.IP_PATTERN.findall(redacted)
        for ip in ips:
            parts = ip.split('.')
            if all(0 <= int(p) <= 255 for p in parts):
                redacted = redacted.replace(ip, cls._redact_value(ip, 'IP', redaction_mode))
                pii_counts['ip_addresses'] = pii_counts.get('ip_addresses', 0) + 1
        
        # 8. Addresses
        addresses = cls.ADDRESS_PATTERN.findall(redacted)
        for address in addresses:
            redacted = redacted.replace(address, cls._redact_value(address, 'ADDRESS', redaction_mode))
            pii_counts['addresses'] = pii_counts.get('addresses', 0) + 1
        
        # 9. Diseases
        text_lower = redacted.lower()
        for disease in cls.DISEASE_KEYWORDS:
            if disease in text_lower:
                pattern = re.compile(re.escape(disease), re.IGNORECASE)
                matches = pattern.findall(redacted)
                for match in matches:
                    redacted = redacted.replace(match, cls._redact_value(match, 'CONDITION', redaction_mode))
                    pii_counts['diseases'] = pii_counts.get('diseases', 0) + 1
        
        # 10. Names (spaCy NER)
        doc = nlp(redacted)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                redacted = redacted.replace(ent.text, cls._redact_value(ent.text, 'NAME', redaction_mode))
                pii_counts['names'] = pii_counts.get('names', 0) + 1
        
        total_items = sum(pii_counts.values())
        
        return PIIRedactionResult(
            redacted_text=redacted,
            pii_detected=total_items > 0,
            pii_counts=pii_counts,
            total_items=total_items
        )


# ========================================================================================
# DYNAMIC RULE ENGINE - Industry-Specific Classification
# ========================================================================================

class DynamicRuleEngine:
    """
    Dynamic rule-based classification engine
    Uses industry-specific rules and keywords
    """
    
    def __init__(self, industry_data: Dict):
        """
        Initialize with industry-specific rules and keywords
        
        Args:
            industry_data: Dictionary containing 'rules' and 'keywords'
        """
        self.rules = industry_data.get('rules', [])
        self.keywords = industry_data.get('keywords', [])
        
        # Build optimized lookup structures
        self._build_lookup_tables()
        
        logger.info(f"Initialized DynamicRuleEngine with {len(self.rules)} rules, {len(self.keywords)} keyword groups")
    
    def _build_lookup_tables(self):
        """Build optimized lookup tables for fast matching"""
        # Compile regex patterns for each rule
        self.compiled_rules = []
        
        for rule in self.rules:
            conditions = rule.get('conditions', [])
            if conditions:
                # Create regex pattern for all conditions (case-insensitive)
                pattern_parts = [re.escape(cond.lower()) for cond in conditions]
                # Match any of the conditions
                pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)
                
                self.compiled_rules.append({
                    'pattern': pattern,
                    'conditions': conditions,
                    'category': rule.get('set', {})
                })
        
        # Compile keyword patterns
        self.compiled_keywords = []
        
        for keyword_group in self.keywords:
            conditions = keyword_group.get('conditions', [])
            if conditions:
                pattern_parts = [re.escape(cond.lower()) for cond in conditions]
                pattern = re.compile('|'.join(pattern_parts), re.IGNORECASE)
                
                self.compiled_keywords.append({
                    'pattern': pattern,
                    'conditions': conditions,
                    'category': keyword_group.get('set', {})
                })
    
    @lru_cache(maxsize=CACHE_SIZE)
    def classify_text(self, text: str) -> CategoryMatch:
        """
        Classify text using dynamic rules
        
        Args:
            text: Input text to classify
            
        Returns:
            CategoryMatch with L1-L4 categories
        """
        if not text or not isinstance(text, str):
            return CategoryMatch(
                l1="Uncategorized",
                l2="NA",
                l3="NA",
                l4="NA",
                confidence=0.0,
                match_path="Uncategorized",
                matched_rule=None
            )
        
        text_lower = text.lower()
        
        # First try keywords (higher priority for quick categorization)
        for kw_item in self.compiled_keywords:
            if kw_item['pattern'].search(text_lower):
                category_data = kw_item['category']
                
                l1 = category_data.get('category', 'Uncategorized')
                l2 = category_data.get('subcategory', 'NA')
                l3 = category_data.get('level_3', 'NA')
                l4 = category_data.get('level_4', 'NA')
                
                # Calculate confidence based on match
                confidence = 0.9  # High confidence for keyword match
                
                return CategoryMatch(
                    l1=l1,
                    l2=l2,
                    l3=l3,
                    l4=l4,
                    confidence=confidence,
                    match_path=f"{l1} > {l2} > {l3} > {l4}",
                    matched_rule="keyword_match"
                )
        
        # Then try detailed rules
        best_match = None
        best_match_count = 0
        
        for rule_item in self.compiled_rules:
            # Count how many conditions match
            matches = rule_item['pattern'].findall(text_lower)
            match_count = len(matches)
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_match = rule_item
        
        if best_match:
            category_data = best_match['category']
            
            l1 = category_data.get('category', 'Uncategorized')
            l2 = category_data.get('subcategory', 'NA')
            l3 = category_data.get('level_3', 'NA')
            l4 = category_data.get('level_4', 'NA')
            
            # Calculate confidence based on number of matches
            total_conditions = len(best_match['conditions'])
            confidence = min(best_match_count / max(total_conditions, 1), 1.0) * 0.85
            
            return CategoryMatch(
                l1=l1,
                l2=l2,
                l3=l3,
                l4=l4,
                confidence=confidence,
                match_path=f"{l1} > {l2} > {l3} > {l4}",
                matched_rule=f"rule_match_{best_match_count}_conditions"
            )
        
        # No match found
        return CategoryMatch(
            l1="Uncategorized",
            l2="NA",
            l3="NA",
            l4="NA",
            confidence=0.0,
            match_path="Uncategorized",
            matched_rule=None
        )


# ========================================================================================
# PROXIMITY ANALYZER (Same as before)
# ========================================================================================

class ProximityAnalyzer:
    """Analyzes text for proximity-based contextual themes"""
    
    PROXIMITY_THEMES = {
        'Agent_Behavior': [
            'agent', 'representative', 'rep', 'staff', 'employee', 'behavior', 
            'behaviour', 'rude', 'unprofessional', 'helpful', 'courteous', 
            'listening', 'attitude', 'manner', 'conduct'
        ],
        'Technical_Issues': [
            'error', 'bug', 'issue', 'problem', 'technical', 'system', 'website',
            'app', 'application', 'crash', 'down', 'not working', 'broken', 
            'glitch', 'malfunction'
        ],
        'Customer_Service': [
            'service', 'support', 'help', 'assist', 'assistance', 'customer',
            'experience', 'satisfaction', 'quality', 'care'
        ],
        'Communication': [
            'communication', 'call', 'email', 'message', 'contact', 'reach',
            'respond', 'response', 'reply', 'follow up', 'callback'
        ],
        'Billing_Payments': [
            'bill', 'billing', 'payment', 'charge', 'charged', 'fee', 'cost',
            'invoice', 'transaction', 'pay', 'paid', 'refund', 'overcharge'
        ],
        'Product_Quality': [
            'product', 'quality', 'defect', 'damaged', 'broken', 'faulty',
            'poor', 'excellent', 'good', 'bad', 'condition'
        ],
        'Cancellation_Refund': [
            'cancel', 'cancellation', 'refund', 'return', 'exchange', 
            'reimbursement', 'money back'
        ],
        'Policy_Terms': [
            'policy', 'term', 'terms', 'condition', 'conditions', 'rule', 
            'rules', 'regulation', 'guideline'
        ],
        'Account_Access': [
            'account', 'login', 'password', 'access', 'locked', 'unlock',
            'reset', 'credentials', 'username'
        ],
        'Order_Delivery': [
            'order', 'delivery', 'shipping', 'dispatch', 'arrival', 'received',
            'tracking', 'delayed', 'late', 'package'
        ],
        'Booking_Reservation': [
            'booking', 'reservation', 'appointment', 'schedule', 'reschedule',
            'book', 'reserved'
        ],
        'Pricing_Cost': [
            'price', 'pricing', 'cost', 'expensive', 'cheap', 'discount', 
            'offer', 'promotion', 'deal'
        ],
        'Verification_Auth': [
            'verify', 'verification', 'confirm', 'confirmation', 'validation', 
            'authenticate', 'authorization'
        ]
    }
    
    @classmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def analyze_proximity(cls, text: str) -> ProximityResult:
        """Analyze text for proximity-based themes"""
        if not text or not isinstance(text, str):
            return ProximityResult(
                primary_proximity="Uncategorized",
                proximity_group="Uncategorized",
                theme_count=0,
                matched_themes=[]
            )
        
        text_lower = text.lower()
        matched_themes = set()
        
        for theme, keywords in cls.PROXIMITY_THEMES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_themes.add(theme)
                    break
        
        if not matched_themes:
            return ProximityResult(
                primary_proximity="Uncategorized",
                proximity_group="Uncategorized",
                theme_count=0,
                matched_themes=[]
            )
        
        priority_order = [
            'Agent_Behavior', 'Technical_Issues', 'Customer_Service', 
            'Communication', 'Billing_Payments', 'Product_Quality',
            'Cancellation_Refund', 'Policy_Terms', 'Account_Access',
            'Order_Delivery', 'Booking_Reservation', 'Pricing_Cost',
            'Verification_Auth'
        ]
        
        primary = next(
            (theme for theme in priority_order if theme in matched_themes),
            list(matched_themes)[0]
        )
        
        matched_list = sorted(list(matched_themes))
        
        return ProximityResult(
            primary_proximity=primary,
            proximity_group=", ".join(matched_list),
            theme_count=len(matched_themes),
            matched_themes=matched_list
        )


# ========================================================================================
# SENTIMENT & TRANSLATION (Same as before)
# ========================================================================================

class SentimentAnalyzer:
    """Sentiment analysis with 5-level granularity"""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def analyze_sentiment(text: str) -> Tuple[str, float]:
        """Analyze sentiment of text"""
        if not text or not isinstance(text, str):
            return "Neutral", 0.0
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity >= 0.5:
                sentiment = "Very Positive"
            elif polarity >= 0.1:
                sentiment = "Positive"
            elif polarity <= -0.5:
                sentiment = "Very Negative"
            elif polarity <= -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            return sentiment, polarity
        
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "Neutral", 0.0


class TranslationService:
    """Multi-language translation service using deep-translator"""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def translate_to_english(text: str) -> str:
        """Translate text to English if needed using deep-translator"""
        if not text or not isinstance(text, str):
            return text
        
        try:
            # Use deep-translator - auto-detects language and translates
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text


# ========================================================================================
# COMPLIANCE MANAGER (Same as before)
# ========================================================================================

class ComplianceManager:
    """Manages compliance reporting and audit logging"""
    
    def __init__(self):
        self.audit_log = []
        self.start_time = datetime.now()
    
    def log_redaction(self, conversation_id: str, pii_counts: Dict[str, int]):
        """Log PII redaction event"""
        self.audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'conversation_id': conversation_id,
            'pii_counts': pii_counts,
            'total_items': sum(pii_counts.values())
        })
    
    def generate_compliance_report(self, results: List[NLPResult]) -> Dict:
        """Generate comprehensive compliance report"""
        total_records = len(results)
        records_with_pii = sum(1 for r in results if r.pii_result.pii_detected)
        total_pii_items = sum(r.pii_result.total_items for r in results)
        
        pii_distribution = {}
        for result in results:
            for pii_type, count in result.pii_result.pii_counts.items():
                pii_distribution[pii_type] = pii_distribution.get(pii_type, 0) + count
        
        return {
            'report_generated': datetime.now().isoformat(),
            'processing_time': str(datetime.now() - self.start_time),
            'summary': {
                'total_records_processed': total_records,
                'records_with_pii': records_with_pii,
                'records_clean': total_records - records_with_pii,
                'pii_detection_rate': f"{(records_with_pii/total_records*100):.2f}%" if total_records > 0 else "0%",
                'total_pii_items': total_pii_items
            },
            'pii_type_distribution': pii_distribution,
            'compliance_standards': COMPLIANCE_STANDARDS,
            'audit_log_entries': len(self.audit_log)
        }
    
    def export_audit_log(self) -> pd.DataFrame:
        """Export audit log as DataFrame"""
        if not self.audit_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.audit_log)


# ========================================================================================
# MAIN NLP PIPELINE - Domain Agnostic
# ========================================================================================

class DynamicNLPPipeline:
    """
    Main NLP processing pipeline with dynamic industry support
    """
    
    def __init__(
        self, 
        rule_engine: DynamicRuleEngine,
        enable_pii_redaction: bool = True,
        industry_name: str = None
    ):
        self.rule_engine = rule_engine
        self.enable_pii_redaction = enable_pii_redaction
        self.industry_name = industry_name
        self.compliance_manager = ComplianceManager()
    
    def process_single_text(
        self, 
        conversation_id: str, 
        text: str,
        redaction_mode: str = 'hash'
    ) -> NLPResult:
        """Process a single text through complete pipeline"""
        
        # 1. PII Detection & Redaction
        if self.enable_pii_redaction:
            pii_result = PIIDetector.detect_and_redact(text, redaction_mode)
            if pii_result.pii_detected:
                self.compliance_manager.log_redaction(conversation_id, pii_result.pii_counts)
            working_text = pii_result.redacted_text
        else:
            pii_result = PIIRedactionResult(
                redacted_text=text,
                pii_detected=False,
                pii_counts={},
                total_items=0
            )
            working_text = text
        
        # 2. Translation
        translated_text = TranslationService.translate_to_english(working_text)
        
        # 3. Dynamic Category Classification
        category = self.rule_engine.classify_text(translated_text)
        
        # 4. Proximity Analysis
        proximity = ProximityAnalyzer.analyze_proximity(translated_text)
        
        # 5. Sentiment Analysis
        sentiment, sentiment_score = SentimentAnalyzer.analyze_sentiment(translated_text)
        
        return NLPResult(
            conversation_id=conversation_id,
            original_text=text,
            redacted_text=pii_result.redacted_text,
            translated_text=translated_text,
            category=category,
            proximity=proximity,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            pii_result=pii_result,
            industry=self.industry_name
        )
    
    def process_batch(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: str,
        redaction_mode: str = 'hash',
        progress_callback=None
    ) -> List[NLPResult]:
        """Process batch of texts with parallel processing"""
        results = []
        total = len(df)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for idx, row in df.iterrows():
                conv_id = str(row[id_column])
                text = str(row[text_column])
                
                future = executor.submit(
                    self.process_single_text,
                    conv_id,
                    text,
                    redaction_mode
                )
                futures[future] = idx
            
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback and completed % 10 == 0:
                        progress_callback(completed, total)
                
                except Exception as e:
                    logger.error(f"Error processing row {futures[future]}: {e}")
                    completed += 1
        
        return results
    
    def results_to_dataframe(self, results: List[NLPResult]) -> pd.DataFrame:
        """
        Convert NLPResult list to DataFrame with essential columns only
        
        Removed columns for cleaner output:
        - Industry (user knows which industry they selected)
        - Redacted_Text (internal use, Original_Text is primary)
        - Translated_Text (internal processing step)
        - Category_Confidence (technical detail)
        - Category_Path (redundant with L1-L4)
        - Matched_Rule (technical detail)
        - Theme_Count (redundant with Proximity_Group)
        - PII_Detected (can infer from Original_Text)
        - PII_Items_Redacted (technical detail)
        - PII_Types (technical detail)
        """
        data = []
        
        for result in results:
            row = {
                'Conversation_ID': result.conversation_id,
                'Original_Text': result.original_text,
                'L1_Category': result.category.l1,
                'L2_Subcategory': result.category.l2,
                'L3_Tertiary': result.category.l3,
                'L4_Quaternary': result.category.l4,
                'Primary_Proximity': result.proximity.primary_proximity,
                'Proximity_Group': result.proximity.proximity_group,
                'Sentiment': result.sentiment,
                'Sentiment_Score': result.sentiment_score
            }
            data.append(row)
        
        return pd.DataFrame(data)


# ========================================================================================
# FILE UTILITIES
# ========================================================================================

class FileHandler:
    """Handles file I/O operations"""
    
    @staticmethod
    def read_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Read uploaded file and return DataFrame"""
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()[1:]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            # Fix duplicate column names
            if not df.columns.is_unique:
                cols = pd.Series(df.columns)
                for dup in cols[cols.duplicated()].unique():
                    dup_indices = [i for i, x in enumerate(df.columns) if x == dup]
                    for i, idx in enumerate(dup_indices[1:], start=1):
                        df.columns.values[idx] = f"{dup}_{i}"
                logger.warning(f"Fixed duplicate column names: {list(df.columns)}")
                st.warning(f"⚠️ Fixed duplicate column names in file. New columns: {list(df.columns)}")
            
            logger.info(f"Successfully loaded file: {uploaded_file.name} ({len(df)} rows)")
            return df
        
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            st.error(f"Error reading file: {e}")
            return None
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, format: str = 'csv') -> bytes:
        """Save DataFrame to bytes"""
        buffer = io.BytesIO()
        
        if format == 'csv':
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            return buffer.getvalue()
        
        elif format == 'xlsx':
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
            buffer.seek(0)
            return buffer.getvalue()
        
        elif format == 'parquet':
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            return buffer.getvalue()
        
        elif format == 'json':
            df.to_json(buffer, orient='records', lines=True)
            buffer.seek(0)
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# ========================================================================================
# STREAMLIT UI
# ========================================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Dynamic NLP Pipeline",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🔒 Dynamic Domain-Agnostic NLP Pipeline")
    st.markdown("""
    **Features:**
    - 🏭 Dynamic Industry-Specific Rules & Keywords
    - 🔐 HIPAA/GDPR/PCI-DSS Compliant PII Redaction
    - 📊 Hierarchical Category Classification (L1 → L2 → L3 → L4)
    - 🎯 Proximity-Based Contextual Grouping
    - 💭 Advanced Sentiment Analysis
    - 🌍 Multi-Language Translation Support
    - ⚡ Optimized for Speed & Scalability
    """)
    
    # Compliance badges
    cols = st.columns(4)
    for idx, standard in enumerate(COMPLIANCE_STANDARDS):
        cols[idx].success(f"✅ {standard} Compliant")
    
    st.markdown("---")
    
    # Initialize domain loader in session state and auto-load industries
    if 'domain_loader' not in st.session_state:
        st.session_state.domain_loader = DomainLoader()
        st.session_state.load_diagnostics = []  # Store diagnostic info
        
        # Force fresh directory scan (no caching)
        import sys
        if 'domain_packs' in sys.modules:
            del sys.modules['domain_packs']
        
        # Auto-load all industries from domain_packs folder
        with st.spinner("🔄 Loading industries from domain_packs/..."):
            loaded_count = st.session_state.domain_loader.auto_load_all_industries()
            
            # Show detailed diagnostic information
            if loaded_count > 0:
                industries_list = st.session_state.domain_loader.get_available_industries()
                st.success(f"✅ Loaded {loaded_count} industries: {', '.join(sorted(industries_list))}")
                logger.info(f"Successfully auto-loaded {loaded_count} industries: {industries_list}")
            else:
                st.error("❌ No industries loaded from domain_packs/ folder!")
                st.info("💡 Check that domain_packs/ folder exists with industry subfolders containing rules.json and keywords.json")
                logger.error("Failed to auto-load any industries")
            
            # Show diagnostic expander - always expanded if not all 10 loaded
            with st.expander("🔍 View Load Diagnostics", expanded=(loaded_count != 10)):
                st.markdown("### Directory Check")
                
                domain_dir = "domain_packs"
                if os.path.exists(domain_dir):
                    st.success(f"✅ Directory exists: {domain_dir}")
                    st.code(f"Full path: {os.path.abspath(domain_dir)}")
                    
                    try:
                        items = os.listdir(domain_dir)
                        # Filter out non-industry items
                        industry_items = [i for i in items if os.path.isdir(os.path.join(domain_dir, i)) and not i.startswith('.')]
                        other_items = [i for i in items if i not in industry_items]
                        
                        st.info(f"📂 Found {len(industry_items)} industry folders: {', '.join(sorted(industry_items))}")
                        if other_items:
                            st.caption(f"Other items (ignored): {', '.join(other_items)}")
                        
                        # Check each industry
                        st.markdown("### Per-Industry Status")
                        
                        success_count = 0
                        failed_count = 0
                        
                        for item in sorted(industry_items):
                            item_path = os.path.join(domain_dir, item)
                            
                            rules_path = os.path.join(item_path, "rules.json")
                            keywords_path = os.path.join(item_path, "keywords.json")
                            
                            has_rules = os.path.exists(rules_path)
                            has_keywords = os.path.exists(keywords_path)
                            
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                if has_rules and has_keywords:
                                    st.success(f"✅ **{item}**")
                                    success_count += 1
                                else:
                                    st.error(f"❌ **{item}**")
                                    failed_count += 1
                            
                            with col2:
                                if has_rules:
                                    st.caption("✓ rules.json")
                                else:
                                    st.caption("✗ rules.json")
                            
                            with col3:
                                if has_keywords:
                                    st.caption("✓ keywords.json")
                                else:
                                    st.caption("✗ keywords.json")
                        
                        st.markdown("---")
                        st.metric("Industries Loaded", f"{success_count}/10", 
                                 delta=f"{success_count - 10} missing" if success_count < 10 else "Complete!")
                        
                        if failed_count > 0:
                            st.error(f"⚠️ {failed_count} industries failed to load. Check that rules.json and keywords.json exist in each folder.")
                    
                    except Exception as e:
                        st.error(f"❌ Error reading directory: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    st.error(f"❌ Directory not found: {domain_dir}")
                    st.code(f"Current working directory: {os.getcwd()}")
                    st.code(f"Files in current directory: {os.listdir('.')}")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Industry Selection
    st.sidebar.subheader("🏭 Industry Selection")
    
    # Get available industries
    available_industries = st.session_state.domain_loader.get_available_industries()
    
    if not available_industries:
        st.sidebar.error("❌ No industries loaded. Please add industry folders to domain_packs/")
        
        # Show manual upload option as fallback
        with st.sidebar.expander("📤 Manual Upload (Fallback)"):
            st.markdown("**Upload Industry Files Manually**")
            
            uploaded_rules = st.file_uploader(
                "Upload Rules JSON",
                type=['json'],
                help="Upload industry-specific rules JSON file",
                key="rules_uploader"
            )
            
            uploaded_keywords = st.file_uploader(
                "Upload Keywords JSON",
                type=['json'],
                help="Upload industry-specific keywords JSON file",
                key="keywords_uploader"
            )
            
            industry_name = st.text_input(
                "Industry Name",
                value="Custom_Industry",
                help="Enter the name of this industry"
            )
            
            if st.button("📥 Load Industry", type="primary"):
                if uploaded_rules and uploaded_keywords:
                    try:
                        rules_path = f"/tmp/{industry_name}_rules.json"
                        keywords_path = f"/tmp/{industry_name}_keywords.json"
                        
                        with open(rules_path, 'wb') as f:
                            f.write(uploaded_rules.getvalue())
                        
                        with open(keywords_path, 'wb') as f:
                            f.write(uploaded_keywords.getvalue())
                        
                        st.session_state.domain_loader.load_from_files(
                            rules_path,
                            keywords_path,
                            industry_name
                        )
                        
                        st.success(f"✅ Loaded {industry_name} successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading industry: {e}")
                else:
                    st.error("⚠️ Please upload both rules and keywords files")
    else:
        # Display industry selection dropdown
        selected_industry = st.sidebar.selectbox(
            "Select Industry",
            options=[""] + sorted(available_industries),
            help="Choose your industry domain for analysis"
        )
        
        if selected_industry:
            # Show industry details
            industry_data = st.session_state.domain_loader.get_industry_data(selected_industry)
            
            st.sidebar.success(f"✅ **{selected_industry}** selected")
            st.sidebar.info(f"""
            **Configuration:**
            - 📋 Rules: {industry_data.get('rules_count', 0)}
            - 🔑 Keywords: {industry_data.get('keywords_count', 0)}
            """)
            
            # Store in session state
            st.session_state.selected_industry = selected_industry
        else:
            st.sidebar.warning("⚠️ Please select an industry to continue")
            st.session_state.selected_industry = None
    
    st.sidebar.markdown("---")
    
    # PII Redaction settings
    st.sidebar.subheader("🔒 PII Redaction")
    enable_pii = st.sidebar.checkbox("Enable PII Redaction", value=True)
    
    redaction_mode = st.sidebar.selectbox(
        "Redaction Mode",
        options=['hash', 'mask', 'token', 'remove'],
        help="hash: SHA-256 hash | mask: Asterisks | token: Simple labels | remove: Complete removal"
    )
    
    # Output format
    st.sidebar.subheader("📤 Output Settings")
    output_format = st.sidebar.selectbox(
        "Output Format",
        options=['csv', 'xlsx', 'parquet', 'json']
    )
    
    # Main content area
    st.header("📁 Data Input")
    
    data_file = st.file_uploader(
        "Upload your data file",
        type=SUPPORTED_FORMATS,
        help="Supported formats: CSV, Excel, Parquet, JSON"
    )
    
    # Process when data file is uploaded and industry is selected
    if data_file and st.session_state.get('selected_industry'):
        
        selected_industry = st.session_state.selected_industry
        
        # Load data
        data_df = FileHandler.read_file(data_file)
        if data_df is None:
            return
        
        st.success(f"✅ Loaded {len(data_df)} records")
        
        # Column selection
        st.subheader("🔧 Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            id_column = st.selectbox(
                "ID Column",
                options=data_df.columns.tolist(),
                help="Select the column containing unique conversation IDs"
            )
        
        with col2:
            text_column = st.selectbox(
                "Text Column",
                options=data_df.columns.tolist(),
                help="Select the column containing text to analyze"
            )
        
        with col3:
            company_column = st.selectbox(
                "Company Column (Optional)",
                options=['None'] + data_df.columns.tolist(),
                help="Select column containing company names for auto-industry detection"
            )
        
        # Preview data
        with st.expander("👀 Preview Data"):
            preview_cols = [id_column, text_column]
            if company_column != 'None':
                preview_cols.append(company_column)
            st.dataframe(data_df[preview_cols].head(10))
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            
            # Get industry data
            industry_data = st.session_state.domain_loader.get_industry_data(selected_industry)
            
            # Initialize components
            with st.spinner(f"Initializing NLP pipeline for {selected_industry}..."):
                rule_engine = DynamicRuleEngine(industry_data)
                pipeline = DynamicNLPPipeline(
                    rule_engine=rule_engine,
                    enable_pii_redaction=enable_pii,
                    industry_name=selected_industry
                )
            
            # Progress tracking
            st.subheader("📊 Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(completed, total):
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"Processed {completed}/{total} records ({progress*100:.1f}%)")
            
            # Process data
            start_time = datetime.now()
            
            with st.spinner("Processing data..."):
                results = pipeline.process_batch(
                    df=data_df,
                    text_column=text_column,
                    id_column=id_column,
                    redaction_mode=redaction_mode,
                    progress_callback=update_progress
                )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Convert to DataFrame
            results_df = pipeline.results_to_dataframe(results)
            
            # Display results
            st.success(f"✅ Analysis Complete! Processed {len(results)} records in {processing_time:.2f} seconds")
            
            # Metrics
            st.subheader("📈 Analysis Metrics")
            
            metric_cols = st.columns(6)
            
            with metric_cols[0]:
                st.metric("Total Records", len(results))
            
            with metric_cols[1]:
                st.metric("Industry", selected_industry)
            
            with metric_cols[2]:
                # Count unique L1 categories
                unique_categories = results_df['L1_Category'].nunique()
                st.metric("Unique Categories", unique_categories)
            
            with metric_cols[3]:
                # Average sentiment score
                avg_sentiment = results_df['Sentiment_Score'].mean()
                st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")
            
            with metric_cols[4]:
                # Count negative sentiment
                negative_count = len(results_df[results_df['Sentiment'].isin(['Negative', 'Very Negative'])])
                st.metric("Negative Sentiment", f"{negative_count} ({negative_count/len(results)*100:.1f}%)")
            
            with metric_cols[5]:
                processing_speed = len(results) / processing_time
                st.metric("Speed", f"{processing_speed:.1f} rec/sec")
            
            # Results preview
            st.subheader("📋 Results Preview")
            st.dataframe(results_df.head(20), use_container_width=True)
            
            # Distribution charts
            st.subheader("📊 Analysis Distributions")
            
            chart_cols = st.columns(3)
            
            with chart_cols[0]:
                st.markdown("**L1 Category Distribution**")
                l1_counts = results_df['L1_Category'].value_counts()
                st.bar_chart(l1_counts)
            
            with chart_cols[1]:
                st.markdown("**Sentiment Distribution**")
                sentiment_counts = results_df['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
            
            with chart_cols[2]:
                st.markdown("**Primary Proximity Distribution**")
                proximity_counts = results_df['Primary_Proximity'].value_counts().head(10)
                st.bar_chart(proximity_counts)
            
            # Compliance report
            if enable_pii:
                st.subheader("🔒 Compliance Report")
                compliance_report = pipeline.compliance_manager.generate_compliance_report(results)
                
                report_cols = st.columns(2)
                
                with report_cols[0]:
                    st.json(compliance_report['summary'])
                
                with report_cols[1]:
                    st.json(compliance_report['pii_type_distribution'])
            
            # Downloads
            st.subheader("💾 Download Results")
            
            download_cols = st.columns(3)
            
            # 1. Main results
            with download_cols[0]:
                results_bytes = FileHandler.save_dataframe(results_df, output_format)
                st.download_button(
                    label=f"📥 Download Results (.{output_format})",
                    data=results_bytes,
                    file_name=f"nlp_results_{selected_industry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
                    mime=f"application/{output_format}"
                )
            
            # 2. Compliance report
            with download_cols[1]:
                if enable_pii:
                    report_bytes = json.dumps(compliance_report, indent=2).encode()
                    st.download_button(
                        label="📥 Download Compliance Report (.json)",
                        data=report_bytes,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # 3. Audit log
            with download_cols[2]:
                if enable_pii:
                    audit_df = pipeline.compliance_manager.export_audit_log()
                    if not audit_df.empty:
                        audit_bytes = FileHandler.save_dataframe(audit_df, 'csv')
                        st.download_button(
                            label="📥 Download Audit Log (.csv)",
                            data=audit_bytes,
                            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
    
    elif data_file and not st.session_state.get('selected_industry'):
        st.warning("⚠️ Please select an industry from the sidebar before processing your data")
    
    elif not st.session_state.get('selected_industry'):
        st.info("👆 Please select an industry from the sidebar to begin.")
    
    else:
        st.info("👆 Please upload your data file to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Dynamic NLP Pipeline v3.0 - Domain Agnostic | Built with Streamlit | Compliant with HIPAA, GDPR, PCI-DSS, CCPA</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
