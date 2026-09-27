"""
generate_dataset.py
--------------------
Builds a labeled dataset of URLs (legitimate=0, phishing=1) and saves both
the raw URLs and their extracted features to CSV.

NOTE ON DATA SOURCE:
This environment has no internet access, so instead of downloading a public
phishing dataset (e.g. PhishTank / UCI Phishing Websites dataset), we
generate a large, rule-based SYNTHETIC dataset that mimics the well-known
structural differences between legitimate and phishing URLs (IP-based
hosts, '@' tricks, suspicious keywords, sketchy TLDs, excessive
subdomains/hyphens, URL shorteners, etc.), with randomness/noise so classes
overlap somewhat -- just like real-world data.

To use a REAL dataset instead: download a CSV (e.g. from Kaggle's
"Phishing Website Detector" or UCI's "Phishing Websites" dataset) with a
'url' and 'label' column, drop it in this data/ folder, and point
train_model.py at it instead of calling this generator.
"""

import random
import string
import sys
import os

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import extract_features  # noqa: E402

random.seed(42)

LEGIT_BRANDS = [
    "google", "wikipedia", "amazon", "github", "microsoft", "apple",
    "netflix", "spotify", "reddit", "linkedin", "nytimes", "bbc",
    "stackoverflow", "python", "mozilla", "cloudflare", "openai",
    "coursera", "khanacademy", "harvard", "nasa", "who", "un",
    "irs", "usps", "dropbox", "adobe", "salesforce", "slack", "zoom"
]

LEGIT_TLDS = ["com", "org", "net", "edu", "gov", "io", "co"]

LEGIT_PATHS = [
    "", "about", "products", "blog/2024/updates", "help/faq",
    "docs/getting-started", "search?q=weather", "news/world",
    "user/profile", "careers", "contact-us", "support/tickets",
    "articles/latest-research", "shop/category/electronics"
]

PHISH_BRAND_TARGETS = [
    "paypal", "amazon", "apple", "microsoft", "netflix", "bankofamerica",
    "wellsfargo", "chase", "facebook", "instagram", "google", "irs",
    "usps", "dhl", "fedex", "outlook", "office365", "icloud", "coinbase",
    "binance", "steam", "linkedin", "americanexpress"
]

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "confirm",
    "signin", "webscr", "password", "billing", "suspend", "unlock",
    "urgent", "security-alert", "recover", "reset-password", "reward"
]

SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "work",
                    "click", "loan", "men", "date", "review", "win"]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly"]


def _rand_str(n, charset=string.ascii_lowercase):
    return "".join(random.choice(charset) for _ in range(n))


def make_legit_url():
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    path = random.choice(LEGIT_PATHS)
    use_www = random.random() < 0.5
    sub = "www." if use_www else ""

    # occasionally a legit-style subdomain e.g. mail.google.com
    if random.random() < 0.15:
        sub = random.choice(["mail.", "docs.", "support.", "shop.", "news."])

    # Realistic noise: a small fraction of legitimate sites still use plain
    # http, have deeper/hyphenated marketing paths, long tracking query
    # strings, or a legit password-reset page containing a "suspicious" word.
    # This overlap is what makes real-world phishing detection non-trivial.
    scheme = "https"
    if random.random() < 0.06:
        scheme = "http"

    if random.random() < 0.08:
        path = f"account/{random.choice(['reset-password','verify-email','update-billing'])}"

    if random.random() < 0.05:
        path = (path + "/" if path else "") + "-".join(_rand_str(4) for _ in range(2))

    url = f"{scheme}://{sub}{brand}.{tld}/{path}".rstrip("/")

    # chance of a query string with normal or slightly long tracking params
    if random.random() < 0.25:
        n_params = random.choice([1, 1, 2, 3])
        parts = [f"{_rand_str(3)}={random.randint(1,99999)}" for _ in range(n_params)]
        url += "?" + "&".join(parts)

    return url


def make_phishing_url():
    style = random.random()

    if style < 0.2:
        # Raw IP address host
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        target = random.choice(PHISH_BRAND_TARGETS)
        word = random.choice(SUSPICIOUS_WORDS)
        url = f"http://{ip}/{target}/{word}.php"

    elif style < 0.35:
        # '@' trick: everything before @ is ignored by browsers
        target = random.choice(PHISH_BRAND_TARGETS)
        fake_host = f"{target}.com.{_rand_str(6)}.com"
        url = f"http://{target}-{random.choice(SUSPICIOUS_WORDS)}@{fake_host}/"

    elif style < 0.55:
        # Brand name buried in a long, hyphenated, subdomain-heavy sketchy domain
        target = random.choice(PHISH_BRAND_TARGETS)
        word = random.choice(SUSPICIOUS_WORDS)
        tld = random.choice(SUSPICIOUS_TLDS)
        sub_count = random.randint(1, 3)
        subs = ".".join(_rand_str(random.randint(3, 8)) for _ in range(sub_count))
        url = (f"http://{subs}.{target}-{word}-{_rand_str(4)}.{tld}/"
               f"{word}/{_rand_str(8)}.html")

    elif style < 0.75:
        # URL shortener hiding the real destination
        shortener = random.choice(SHORTENERS)
        url = f"http://{shortener}/{_rand_str(7, string.ascii_letters + string.digits)}"

    else:
        # Long, messy query string + suspicious keyword + double slash redirect trick
        target = random.choice(PHISH_BRAND_TARGETS)
        word = random.choice(SUSPICIOUS_WORDS)
        tld = random.choice(SUSPICIOUS_TLDS + ["com"])
        url = (f"http://{target}{_rand_str(5)}.{tld}//{word}"
               f"?user={_rand_str(6)}&token={_rand_str(12, string.ascii_letters+string.digits)}"
               f"&redirect={_rand_str(5)}.com")

    # Phishing sites increasingly buy real HTTPS certs (Let's Encrypt etc.),
    # so make this common rather than rare -- it removes an overly-easy signal.
    if random.random() < 0.55:
        url = url.replace("http://", "https://", 1)

    # A slice of harder phishing cases: short, clean-looking domain with a
    # near-miss brand name and no obvious keyword -- mimics real typosquats.
    if random.random() < 0.1:
        target = random.choice(PHISH_BRAND_TARGETS)
        typo = target[:-1] + random.choice(string.ascii_lowercase) if len(target) > 3 else target + "a"
        tld = random.choice(["com", "net", "co"])
        url = f"https://{typo}.{tld}/{random.choice(LEGIT_PATHS)}".rstrip("/")

    return url


def build_dataset(n_per_class=2500, label_noise=0.02):
    rows = []
    for _ in range(n_per_class):
        rows.append([make_legit_url(), 0])
    for _ in range(n_per_class):
        rows.append([make_phishing_url(), 1])

    # Inject a small amount of label noise to mimic real-world dataset
    # imperfections (mislabeled samples happen even in curated datasets).
    for row in rows:
        if random.random() < label_noise:
            row[1] = 1 - row[1]

    random.shuffle(rows)
    return [tuple(r) for r in rows]


def main():
    rows = build_dataset(n_per_class=2500)

    records = []
    for url, label in rows:
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = label
        records.append(feats)

    df = pd.DataFrame(records)
    # reorder: url first, features, label last
    cols = ["url"] + [c for c in df.columns if c not in ("url", "label")] + ["label"]
    df = df[cols]

    out_path = os.path.join(os.path.dirname(__file__), "phishing_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
