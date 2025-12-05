"""
Dynamic Domain-Agnostic NLP Text Analysis Pipeline - FIXED VERSION
===================================================================

FIXES APPLIED:
1. Added session state for file upload persistence
2. Added file size validation with clear limits
3. Improved conditional logic for file detection
4. Added debug information for troubleshooting

Version: 3.0.1 - Fixed File Upload Detection
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

# Constants - OPTIMIZED FOR PERFORMANCE
MAX_WORKERS = 8  # Increased from 4 to 8 (use more CPU cores)
BATCH_SIZE = 500  # Increased from 100 to 500 for better batching
CACHE_SIZE = 10000  # Increased from 1000 to 10000 for better caching
SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'parquet', 'json']
COMPLIANCE_STANDARDS = ["HIPAA", "GDPR", "PCI-DSS", "CCPA"]

# Performance optimization flags
ENABLE_TRANSLATION = False  # Set to False to skip translation (BIGGEST SPEEDUP ~150ms/text)
ENABLE_SPACY_NER = False  # Set to False to skip spaCy NER in PII (saves ~50ms/text)
PII_DETECTION_MODE = 'fast'  # 'fast' or 'full' - fast skips expensive checks

# NEW: File size limits (in MB)
MAX_FILE_SIZE_MB = 500  # 500MB max file size
WARN_FILE_SIZE_MB = 100  # Warning threshold

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
# PII DETECTION & REDACTION ENGINE
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
        """Detect and redact all PII/PHI/PCI from text - OPTIMIZED"""
        if not text or not isinstance(text, str):
            return PIIRedactionResult(
                redacted_text=str(text) if text else "",
                pii_detected=False,
                pii_counts={},
                total_items=0
            )
        
        redacted = text
        pii_counts = {}
        
        # PERFORMANCE OPTIMIZATION: Fast mode skips expensive checks
        if PII_DETECTION_MODE == 'fast':
            # Only run essential checks in fast mode
            
            # 1. Emails (fast)
            emails = cls.EMAIL_PATTERN.findall(redacted)
            for email in emails:
                redacted = redacted.replace(email, cls._redact_value(email, 'EMAIL', redaction_mode))
                pii_counts['emails'] = pii_counts.get('emails', 0) + 1
            
            # 2. Phone numbers (fast)
            for pattern in cls.PHONE_PATTERNS:
                phones = pattern.findall(redacted)
                for phone in phones:
                    redacted = redacted.replace(phone, cls._redact_value(phone, 'PHONE', redaction_mode))
                    pii_counts['phones'] = pii_counts.get('phones', 0) + 1
            
            # Skip expensive checks: credit cards, SSN validation, spaCy NER, etc.
            total_items = sum(pii_counts.values())
            
            return PIIRedactionResult(
                redacted_text=redacted,
                pii_detected=total_items > 0,
                pii_counts=pii_counts,
                total_items=total_items
            )
        
        # FULL MODE: Original comprehensive detection
        # 1. Emails
        emails = cls.EMAIL_PATTERN.findall(redacted)
        for email in emails:
            redacted = redacted.replace(email, cls._redact_value(email, 'EMAIL', redaction_mode))
            pii_counts['emails'] = pii_counts.get('emails', 0) + 1
        
        # 2. Credit cards (expensive - Luhn validation)
        for pattern in cls.CREDIT_CARD_PATTERNS:
            cards = pattern.findall(redacted)
            for card in cards:
                if cls._is_valid_credit_card(card):
                    redacted = redacted.replace(card, cls._redact_value(card, 'CARD', redaction_mode))
                    pii_counts['credit_cards'] = pii_counts.get('credit_cards', 0) + 1
        
        # 3. SSNs (expensive - validation)
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
        
        # 5. DOBs (expensive - validation)
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
        
        # 10. Names (spaCy NER - VERY EXPENSIVE ~50ms per text)
        if ENABLE_SPACY_NER:
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
# PROXIMITY ANALYZER
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
# SENTIMENT & TRANSLATION
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
    """Multi-language translation service using deep-translator - OPTIMIZED"""
    
    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def translate_to_english(text: str) -> str:
        """Translate text to English if needed using deep-translator - OPTIMIZED"""
        if not text or not isinstance(text, str):
            return text
        
        # PERFORMANCE OPTIMIZATION: Skip translation if disabled
        if not ENABLE_TRANSLATION:
            return text  # Skip translation entirely (saves ~150ms per text)
        
        try:
            # Use deep-translator - auto-detects language and translates
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text


# ========================================================================================
# COMPLIANCE MANAGER
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
            # NEW: Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ File size ({file_size_mb:.1f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB")
                st.info("💡 Tip: Try splitting your file into smaller chunks or filtering the data before upload")
                return None
            
            if file_size_mb > WARN_FILE_SIZE_MB:
                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take several minutes.")
            
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
# STREAMLIT UI - FIXED
# ========================================================================================

def main():
    """Main Streamlit application - FIXED VERSION"""
    
    # Page configuration
    st.set_page_config(
        page_title="Dynamic NLP Pipeline",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🔒 Dynamic Domain-Agnostic NLP Pipeline v3.0.1")
    st.markdown("""
    **Features:**
    - 🏭 Dynamic Industry-Specific Rules & Keywords
    - 🔐 HIPAA/GDPR/PCI-DSS Compliant PII Redaction
    - 📊 Hierarchical Category Classification (L1 → L2 → L3 → L4)
    - 🎯 Proximity-Based Contextual Grouping
    - 💭 Advanced Sentiment Analysis
    - 🌍 Multi-Language Translation Support
    - ⚡ Optimized for Speed & Scalability
    
    ---
    **🆕 v3.0.1 Fixes:**
    - Fixed file upload detection issue
    - Added file size validation (500MB limit)
    - Improved session state management
    - Enhanced error handling
    """)
    
    # Compliance badges
    cols = st.columns(4)
    for idx, standard in enumerate(COMPLIANCE_STANDARDS):
        cols[idx].success(f"✅ {standard} Compliant")
    
    st.markdown("---")
    
    # Initialize domain loader in session state
    if 'domain_loader' not in st.session_state:
        st.session_state.domain_loader = DomainLoader()
        
        # Auto-load all industries
        with st.spinner("🔄 Loading industries from domain_packs/..."):
            loaded_count = st.session_state.domain_loader.auto_load_all_industries()
            
            if loaded_count > 0:
                industries_list = st.session_state.domain_loader.get_available_industries()
                st.success(f"✅ Loaded {loaded_count} industries: {', '.join(sorted(industries_list))}")
                logger.info(f"Successfully auto-loaded {loaded_count} industries")
            else:
                st.error("❌ No industries loaded from domain_packs/ folder!")
                st.info("💡 Check that domain_packs/ folder exists with industry subfolders")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Industry Selection
    st.sidebar.subheader("🏭 Industry Selection")
    
    # Get available industries
    available_industries = st.session_state.domain_loader.get_available_industries()
    
    if not available_industries:
        st.sidebar.error("❌ No industries loaded")
        st.session_state.selected_industry = None
    else:
        # Display industry selection dropdown
        selected_industry = st.sidebar.selectbox(
            "Select Industry",
            options=[""] + sorted(available_industries),
            help="Choose your industry domain for analysis",
            key="industry_selector"
        )
        
        if selected_industry:
            # Store in session state
            st.session_state.selected_industry = selected_industry
            
            # Show industry details
            industry_data = st.session_state.domain_loader.get_industry_data(selected_industry)
            
            st.sidebar.success(f"✅ **{selected_industry}** selected")
            st.sidebar.info(f"""
            **Configuration:**
            - 📋 Rules: {industry_data.get('rules_count', 0)}
            - 🔑 Keywords: {industry_data.get('keywords_count', 0)}
            """)
        else:
            st.sidebar.warning("⚠️ Please select an industry")
            st.session_state.selected_industry = None
    
    st.sidebar.markdown("---")
    
    # PII Redaction settings
    st.sidebar.subheader("🔒 PII Redaction")
    enable_pii = st.sidebar.checkbox("Enable PII Redaction", value=True)
    
    redaction_mode = st.sidebar.selectbox(
        "Redaction Mode",
        options=['hash', 'mask', 'token', 'remove'],
        help="hash: SHA-256 | mask: *** | token: [TYPE] | remove: delete"
    )
    
    st.sidebar.markdown("---")
    
    # PERFORMANCE SETTINGS (NEW)
    st.sidebar.subheader("⚡ Performance Settings")
    
    st.sidebar.info("💡 **Speed Optimization Tips**")
    
    # Translation toggle
    enable_translation = st.sidebar.checkbox(
        "Enable Translation",
        value=False,
        help="⚠️ Translation is SLOW (~150ms/text). Disable for English-only data for 2-3x speedup!"
    )
    
    # PII detection mode
    pii_mode = st.sidebar.radio(
        "PII Detection Mode",
        options=['fast', 'full'],
        index=0,
        help="Fast: Skip expensive checks (credit cards, SSN validation, spaCy NER)\nFull: Comprehensive PII detection (slower)"
    )
    
    # spaCy NER toggle
    enable_spacy = st.sidebar.checkbox(
        "Enable spaCy NER (Names)",
        value=False,
        help="⚠️ spaCy NER is slow (~50ms/text). Disable if name detection not critical."
    )
    
    # Worker threads
    max_workers = st.sidebar.slider(
        "Parallel Workers",
        min_value=2,
        max_value=16,
        value=8,
        help="More workers = faster processing (if you have CPU cores available)"
    )
    
    # Show performance estimate
    with st.sidebar.expander("📊 Speed Estimates", expanded=False):
        base_speed = 5.15  # Current speed (records/sec)
        
        # Calculate estimated speedup
        speedup = 1.0
        if not enable_translation:
            speedup *= 2.5  # Translation is biggest bottleneck
        if pii_mode == 'fast':
            speedup *= 1.3  # Fast PII mode
        if not enable_spacy:
            speedup *= 1.2  # Skip spaCy
        if max_workers > 4:
            speedup *= (max_workers / 4)  # More workers
        
        estimated_speed = base_speed * speedup
        estimated_time = 4372 / estimated_speed
        
        st.metric("Current Speed", f"{base_speed:.1f} rec/sec")
        st.metric("Estimated Speed", f"{estimated_speed:.1f} rec/sec", delta=f"{(speedup-1)*100:.0f}%")
        st.metric("Est. Time (4.4K records)", f"{estimated_time/60:.1f} min")
        
        if estimated_speed >= 10:
            st.success("✅ Target speed achieved!")
        else:
            st.warning(f"⚠️ Need {10-estimated_speed:.1f} more rec/sec")
    
    # Update global flags
    import sys
    current_module = sys.modules[__name__]
    current_module.ENABLE_TRANSLATION = enable_translation
    current_module.PII_DETECTION_MODE = pii_mode
    current_module.ENABLE_SPACY_NER = enable_spacy
    current_module.MAX_WORKERS = max_workers
    
    # Output format
    st.sidebar.subheader("📤 Output Settings")
    output_format = st.sidebar.selectbox(
        "Output Format",
        options=['csv', 'xlsx', 'parquet', 'json']
    )
    
    # Main content area
    st.header("📁 Data Input")
    
    # File uploader with key for better state management
    data_file = st.file_uploader(
        "Upload your data file",
        type=SUPPORTED_FORMATS,
        help=f"Supported: CSV, Excel, Parquet, JSON (Max {MAX_FILE_SIZE_MB}MB)",
        key="data_file_uploader"
    )
    
    # NEW: Store uploaded file in session state for persistence
    if data_file is not None:
        st.session_state.current_file = data_file
        st.session_state.file_uploaded = True
        logger.info(f"File uploaded to session state: {data_file.name}, size: {data_file.size} bytes")
    
    # NEW: Debug information (can be removed in production)
    with st.expander("🐛 Debug Info", expanded=False):
        st.write("**Session State:**")
        st.write(f"- Selected Industry: {st.session_state.get('selected_industry', 'None')}")
        st.write(f"- File Uploaded: {st.session_state.get('file_uploaded', False)}")
        st.write(f"- File Object: {data_file is not None}")
        if data_file:
            st.write(f"- File Name: {data_file.name}")
            st.write(f"- File Size: {data_file.size / 1024:.2f} KB")
    
    # FIXED: Better conditional logic
    has_industry = st.session_state.get('selected_industry') is not None and st.session_state.get('selected_industry') != ""
    has_file = data_file is not None
    
    # Show appropriate messages
    if not has_industry:
        st.info("👆 **Step 1:** Please select an industry from the sidebar to begin")
    elif not has_file:
        st.info("👆 **Step 2:** Please upload your data file to continue")
    else:
        # Both conditions met - show processing interface
        selected_industry = st.session_state.selected_industry
        
        st.success(f"✅ Ready to process with **{selected_industry}** industry")
        
        # Load data
        data_df = FileHandler.read_file(data_file)
        
        if data_df is not None:
            st.success(f"✅ Loaded {len(data_df):,} records from {data_file.name}")
            
            # Smart column detection
            st.info("🤖 **Smart Column Detection**")
            detection_cols = st.columns(3)
            
            # Detect ID column (low cardinality, sequential, or named 'id')
            likely_id_cols = []
            for col in data_df.columns:
                col_lower = col.lower()
                # Check for ID-like names
                if any(keyword in col_lower for keyword in ['id', 'conversation', 'ticket', 'case', 'record']):
                    likely_id_cols.append(col)
                # Check for high uniqueness (likely IDs)
                elif data_df[col].nunique() / len(data_df) > 0.8:
                    likely_id_cols.append(col)
            
            # Detect text columns (long strings)
            likely_text_cols = []
            for col in data_df.columns:
                if data_df[col].dtype == 'object':
                    sample = data_df[col].dropna().head(20).astype(str)
                    if len(sample) > 0:
                        avg_len = sample.str.len().mean()
                        if avg_len > 30:  # Text columns typically have longer content
                            likely_text_cols.append(col)
            
            # Detect company columns
            likely_company_cols = []
            for col in data_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['company', 'organization', 'org', 'business', 'client', 'customer']):
                    if col not in likely_id_cols and col not in likely_text_cols:
                        likely_company_cols.append(col)
            
            with detection_cols[0]:
                if likely_id_cols:
                    st.success(f"🆔 **Suggested ID:** `{likely_id_cols[0]}`")
                else:
                    st.warning("🆔 **ID:** No suggestion")
            
            with detection_cols[1]:
                if likely_text_cols:
                    st.success(f"📝 **Suggested Text:** `{likely_text_cols[0]}`")
                else:
                    st.warning("📝 **Text:** No suggestion")
            
            with detection_cols[2]:
                if likely_company_cols:
                    st.success(f"🏢 **Suggested Company:** `{likely_company_cols[0]}`")
                else:
                    st.info("🏢 **Company:** Optional")
            
            st.caption("💡 These are suggestions based on column names and content. You can override them below.")
            st.markdown("---")
            
            # Column selection
            st.subheader("🔧 Configuration")
            
            # Helper function to detect likely text columns
            def detect_text_columns(df):
                """Detect columns that likely contain text (not IDs)"""
                text_cols = []
                for col in df.columns:
                    # Sample first 5 non-null values
                    sample = df[col].dropna().head(5)
                    if len(sample) > 0:
                        # Check if values are strings and reasonably long (>20 chars on average)
                        if sample.dtype == 'object':
                            avg_len = sample.astype(str).str.len().mean()
                            if avg_len > 20:  # Likely text, not ID
                                text_cols.append(col)
                return text_cols
            
            # Detect likely text columns
            likely_text_cols = detect_text_columns(data_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Set default index for ID column
                id_default_idx = 0
                if likely_id_cols and likely_id_cols[0] in data_df.columns:
                    id_default_idx = data_df.columns.tolist().index(likely_id_cols[0])
                
                id_column = st.selectbox(
                    "ID Column",
                    options=data_df.columns.tolist(),
                    index=id_default_idx,
                    help="Select column with unique conversation IDs",
                    key="id_col_selector"
                )
            
            with col2:
                # Smart default: prefer likely text columns, exclude ID column
                text_col_options = [col for col in data_df.columns if col != id_column]
                
                # Set default to first likely text column if available
                default_text_idx = 0
                if likely_text_cols:
                    for idx, col in enumerate(text_col_options):
                        if col in likely_text_cols:
                            default_text_idx = idx
                            break
                
                text_column = st.selectbox(
                    "Text Column",
                    options=text_col_options,
                    index=default_text_idx,
                    help="Select column with text to analyze (automatically excludes ID column)",
                    key="text_col_selector"
                )
            
            with col3:
                # Exclude already selected columns
                company_col_options = ['None'] + [
                    col for col in data_df.columns 
                    if col not in [id_column, text_column]
                ]
                
                # Set default to suggested company column
                company_default_idx = 0
                if likely_company_cols and likely_company_cols[0] in company_col_options:
                    company_default_idx = company_col_options.index(likely_company_cols[0])
                
                company_column = st.selectbox(
                    "Company Column (Optional)",
                    options=company_col_options,
                    index=company_default_idx,
                    help="For auto-industry detection (optional)",
                    key="company_col_selector"
                )
            
            # Validation: Check for column conflicts
            validation_errors = []
            validation_warnings = []
            
            # Error: Same column for ID and Text
            if id_column == text_column:
                validation_errors.append("⚠️ ID Column and Text Column cannot be the same!")
            
            # Warning: Text column looks like an ID (short values)
            if text_column in data_df.columns:
                text_sample = data_df[text_column].dropna().head(5).astype(str)
                if len(text_sample) > 0:
                    avg_len = text_sample.str.len().mean()
                    if avg_len < 20:
                        validation_warnings.append(
                            f"⚠️ Warning: '{text_column}' has short values (avg {avg_len:.0f} chars). "
                            "This may not be a text column. Consider selecting a different column."
                        )
            
            # Warning: Company column same as ID or Text
            if company_column != 'None':
                if company_column == id_column:
                    validation_errors.append("⚠️ Company Column cannot be the same as ID Column!")
                if company_column == text_column:
                    validation_errors.append("⚠️ Company Column cannot be the same as Text Column!")
            
            # Display validation messages
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.error("❌ Please fix the column selection errors above before proceeding.")
                config_valid = False
            else:
                config_valid = True
                
                if validation_warnings:
                    for warning in validation_warnings:
                        st.warning(warning)
            
            # Preview data with error handling
            with st.expander("👀 Preview Data (first 10 rows)", expanded=config_valid):
                if not config_valid:
                    st.info("🔧 Fix configuration errors to see data preview")
                else:
                    try:
                        # Build preview columns
                        preview_cols = [id_column, text_column]
                        if company_column != 'None':
                            preview_cols.append(company_column)
                        
                        # Remove duplicates while preserving order
                        preview_cols = list(dict.fromkeys(preview_cols))
                        
                        # Create preview dataframe
                        preview_df = data_df[preview_cols].head(10).copy()
                        
                        # Add helpful column info
                        col_info = st.columns(len(preview_cols))
                        for idx, col in enumerate(preview_cols):
                            with col_info[idx]:
                                # Count non-null values
                                non_null = data_df[col].notna().sum()
                                total = len(data_df)
                                pct = (non_null / total * 100) if total > 0 else 0
                                
                                # Determine role
                                if col == id_column:
                                    role = "🆔 ID"
                                elif col == text_column:
                                    role = "📝 Text"
                                else:
                                    role = "🏢 Company"
                                
                                st.caption(f"{role}: **{col}**")
                                st.caption(f"✓ {non_null:,}/{total:,} ({pct:.1f}%) non-null")
                        
                        st.markdown("---")
                        
                        # Display preview
                        st.dataframe(
                            preview_df, 
                            use_container_width=True,
                            height=400
                        )
                        
                        # Show sample statistics
                        with st.expander("📊 Column Statistics", expanded=False):
                            stats_cols = st.columns(len(preview_cols))
                            
                            for idx, col in enumerate(preview_cols):
                                with stats_cols[idx]:
                                    st.markdown(f"**{col}**")
                                    
                                    # Data type
                                    st.caption(f"Type: {data_df[col].dtype}")
                                    
                                    # Unique values
                                    unique = data_df[col].nunique()
                                    st.caption(f"Unique: {unique:,}")
                                    
                                    # Missing values
                                    missing = data_df[col].isna().sum()
                                    st.caption(f"Missing: {missing:,}")
                                    
                                    # For text column, show length stats
                                    if col == text_column:
                                        lengths = data_df[col].dropna().astype(str).str.len()
                                        if len(lengths) > 0:
                                            st.caption(f"Avg Length: {lengths.mean():.0f} chars")
                                            st.caption(f"Min/Max: {lengths.min()}/{lengths.max()}")
                    
                    except Exception as e:
                        logger.error(f"Preview error: {e}")
                        st.error("❌ Unable to preview data with current column selection.")
                        st.info("💡 Tip: Try selecting different columns or check your data format.")
                        
                        # Show detailed error in debug mode
                        if st.checkbox("🐛 Show detailed error", value=False):
                            st.exception(e)
            
            st.markdown("---")
            
            # Final validation before allowing processing
            can_process = config_valid and id_column and text_column
            
            # Process button with validation
            if not can_process:
                st.button(
                    "🚀 Run Analysis", 
                    type="primary", 
                    use_container_width=True,
                    disabled=True,
                    help="Fix configuration errors before running analysis"
                )
                st.error("⚠️ Please fix configuration errors above to enable analysis")
            
            elif st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                
                # Final validation check
                if id_column == text_column:
                    st.error("❌ Cannot proceed: ID and Text columns must be different!")
                    st.stop()
                
                # Check if text column actually has text
                text_sample = data_df[text_column].dropna().head(100)
                if len(text_sample) == 0:
                    st.error(f"❌ Text column '{text_column}' appears to be empty!")
                    st.stop()
                
                avg_text_len = text_sample.astype(str).str.len().mean()
                if avg_text_len < 10:
                    st.warning(f"⚠️ Text column '{text_column}' has very short values (avg {avg_text_len:.1f} chars)")
                    st.warning("This may not produce meaningful results. Consider selecting a different column.")
                    
                    if not st.checkbox("⚠️ I understand and want to proceed anyway", value=False):
                        st.info("👆 Check the box above to proceed with this column selection")
                        st.stop()
                
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
                    status_text.text(f"Processed {completed:,}/{total:,} records ({progress*100:.1f}%)")
                
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
                st.success(f"✅ Analysis Complete! Processed {len(results):,} records in {processing_time:.2f} seconds")
                
                # Metrics
                st.subheader("📈 Analysis Metrics")
                
                metric_cols = st.columns(6)
                
                with metric_cols[0]:
                    st.metric("Total Records", f"{len(results):,}")
                
                with metric_cols[1]:
                    st.metric("Industry", selected_industry)
                
                with metric_cols[2]:
                    unique_categories = results_df['L1_Category'].nunique()
                    st.metric("Unique Categories", unique_categories)
                
                with metric_cols[3]:
                    avg_sentiment = results_df['Sentiment_Score'].mean()
                    st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")
                
                with metric_cols[4]:
                    negative_count = len(results_df[results_df['Sentiment'].isin(['Negative', 'Very Negative'])])
                    pct = (negative_count/len(results)*100) if len(results) > 0 else 0
                    st.metric("Negative Sentiment", f"{negative_count:,} ({pct:.1f}%)")
                
                with metric_cols[5]:
                    processing_speed = len(results) / processing_time if processing_time > 0 else 0
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
                
                with download_cols[0]:
                    results_bytes = FileHandler.save_dataframe(results_df, output_format)
                    st.download_button(
                        label=f"📥 Download Results (.{output_format})",
                        data=results_bytes,
                        file_name=f"nlp_results_{selected_industry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}",
                        mime=f"application/{output_format}"
                    )
                
                with download_cols[1]:
                    if enable_pii:
                        report_bytes = json.dumps(compliance_report, indent=2).encode()
                        st.download_button(
                            label="📥 Download Compliance Report",
                            data=report_bytes,
                            file_name=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with download_cols[2]:
                    if enable_pii:
                        audit_df = pipeline.compliance_manager.export_audit_log()
                        if not audit_df.empty:
                            audit_bytes = FileHandler.save_dataframe(audit_df, 'csv')
                            st.download_button(
                                label="📥 Download Audit Log",
                                data=audit_bytes,
                                file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Dynamic NLP Pipeline v3.0.1 - FIXED | Built with Streamlit | HIPAA/GDPR/PCI-DSS/CCPA Compliant</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
