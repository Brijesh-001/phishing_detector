"""
features.py
------------
Central feature-extraction logic for the Phishing URL Detector.

IMPORTANT: This exact module is used both when building the training
dataset and inside the Streamlit app at prediction time, so the model
always sees features computed the same way.
"""

import re
from urllib.parse import urlparse

# Keywords commonly stuffed into phishing URLs to create false urgency/trust
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank", "confirm",
    "signin", "sign-in", "webscr", "password", "billing", "invoice",
    "suspend", "unlock", "urgent", "click", "free", "gift", "bonus",
    "limited", "security", "alert", "recover", "reset"
]

# Domains known for free/very-cheap URL shortening (often abused in phishing)
SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly"
]

# TLDs that are free / very cheap to register and disproportionately abused
SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan",
    "gq", "men", "date", "racing", "review", "win"
]

IP_PATTERN = re.compile(
    r"^(?:http[s]?://)?"
    r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
)


FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "has_ip",
    "has_at_symbol",
    "has_https",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_subdomains",
    "has_suspicious_word",
    "has_shortener",
    "has_suspicious_tld",
    "has_double_slash_redirect",
    "num_query_params",
    "has_port",
    "digit_letter_ratio",
]


def _safe_urlparse(url: str):
    """Ensure urlparse has a scheme to work with so netloc/path parse correctly."""
    if not re.match(r"^[a-zA-Z]+://", url):
        # No scheme given -> assume http for parsing purposes only
        parsed = urlparse("http://" + url)
    else:
        parsed = urlparse(url)
    return parsed


def extract_features(url: str) -> dict:
    """Extract a fixed-order dict of numeric features from a raw URL string."""
    url = (url or "").strip()
    parsed = _safe_urlparse(url)

    domain = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Strip port from domain for keyword/tld checks, but remember if it existed
    has_port = 1 if ":" in domain and not domain.startswith("[") else 0
    domain_no_port = domain.split(":")[0]

    url_lower = url.lower()

    features = {}
    features["url_length"] = len(url)
    features["domain_length"] = len(domain_no_port)
    features["path_length"] = len(path)

    features["has_ip"] = 1 if IP_PATTERN.match(url) or re.match(
        r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain_no_port
    ) else 0

    features["has_at_symbol"] = 1 if "@" in url else 0
    features["has_https"] = 1 if parsed.scheme == "https" else 0

    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_digits"] = sum(c.isdigit() for c in url)

    # crude subdomain count: pieces in domain minus (sld + tld)
    domain_parts = [p for p in domain_no_port.split(".") if p]
    features["num_subdomains"] = max(len(domain_parts) - 2, 0)

    features["has_suspicious_word"] = 1 if any(
        w in url_lower for w in SUSPICIOUS_WORDS
    ) else 0

    features["has_shortener"] = 1 if any(
        s in domain_no_port for s in SHORTENER_DOMAINS
    ) else 0

    tld = domain_parts[-1] if domain_parts else ""
    features["has_suspicious_tld"] = 1 if tld in SUSPICIOUS_TLDS else 0

    # "//" appearing anywhere in the path after position 7 often signals a
    # redirect trick (e.g. http://legit.com//evil.com)
    features["has_double_slash_redirect"] = 1 if url.find("//", 8) != -1 else 0

    features["num_query_params"] = len(query.split("&")) if query else 0
    features["has_port"] = has_port

    letters = sum(c.isalpha() for c in url)
    features["digit_letter_ratio"] = round(
        features["num_digits"] / letters, 4
    ) if letters > 0 else 0.0

    # Return in fixed order
    return {name: features[name] for name in FEATURE_NAMES}


def extract_features_batch(urls):
    """Vectorized-ish convenience wrapper returning a list of feature dicts."""
    return [extract_features(u) for u in urls]
