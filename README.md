# URL Forensics — Phishing Website Detection

A complete, self-contained machine learning project that classifies URLs as
**phishing** or **legitimate** based purely on structural features (length,
`@` symbols, HTTPS usage, suspicious keywords, IP hosts, URL shorteners,
etc.), plus a monochrome Streamlit front end to explore it interactively.

```
phishing_detector/
├── app.py                     # Streamlit front end
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # theme config
├── data/
│   ├── generate_dataset.py    # builds the synthetic labeled dataset
│   └── phishing_dataset.csv   # generated dataset (5,000 rows)
├── src/
│   ├── features.py            # shared feature-extraction logic
│   └── train_model.py         # trains + evaluates the model
└── model/
    ├── phishing_model.pkl     # trained RandomForestClassifier
    ├── feature_columns.pkl    # exact feature order the model expects
    └── metrics.json           # accuracy / precision / recall / confusion matrix
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the dataset and retrain the model
python data/generate_dataset.py
python src/train_model.py

# 3. Launch the app
streamlit run app.py
```

The repo already ships with a trained model and dataset, so step 2 is
optional — you can go straight to `streamlit run app.py`.

## How it works

### 1. Features (`src/features.py`)

Every URL is converted into 18 numeric signals used by both training and
the live app, so predictions are always consistent:

| Feature | What it captures |
|---|---|
| `url_length`, `domain_length`, `path_length` | Raw lengths — phishing URLs tend to run long |
| `has_ip` | Host is a raw IP address instead of a domain name |
| `has_at_symbol` | Classic `@` trick (browsers ignore everything before it) |
| `has_https` | Whether the connection is encrypted |
| `num_dots`, `num_hyphens`, `num_underscores`, `num_digits` | Character-level noise, often elevated in phishing |
| `num_subdomains` | Deeply nested subdomains used to bury the real domain |
| `has_suspicious_word` | Keywords like `login`, `verify`, `secure`, `account`, `confirm`... |
| `has_shortener` | Uses bit.ly / tinyurl / etc. to hide the destination |
| `has_suspicious_tld` | Cheap/free TLDs disproportionately abused (`.tk`, `.xyz`, `.top`...) |
| `has_double_slash_redirect` | `//` appearing mid-path, a known redirect trick |
| `num_query_params` | Number of `&`-separated query parameters |
| `has_port` | Non-standard port explicitly specified |
| `digit_letter_ratio` | Digits relative to letters — spammy generated hostnames skew high |

### 2. Dataset (`data/generate_dataset.py`)

This sandbox has no internet access, so instead of downloading a live feed
(e.g. PhishTank) or a static Kaggle/UCI CSV, the dataset is **synthetically
generated** from rules that mirror real phishing structure:

- Legitimate URLs: real-looking brand names, standard TLDs, normal paths,
  occasional query strings — with some realistic noise (a few plain-`http`
  legitimate pages, password-reset paths that happen to contain "suspicious"
  words, longer tracking query strings).
- Phishing URLs: five different attack styles — raw IP hosts, `@`-symbol
  tricks, hyphen/subdomain-stuffed sketchy domains, URL shorteners, and
  messy redirect/query-string patterns — plus typosquats and a chance of
  a legitimate-looking HTTPS cert (since real phishing sites increasingly
  buy real certificates).
- 2% random label noise is injected to mimic imperfections in real-world
  labeled datasets.

This keeps classes realistically overlapping (~95% test accuracy) rather
than trivially separable (100%).

**To use a real dataset instead:** drop a CSV with `url` and `label`
columns (0 = legitimate, 1 = phishing) into `data/`, point
`src/train_model.py`'s `DATA_PATH` at it, and re-run training. Good public
sources: Kaggle's "Phishing Website Detector" dataset, or UCI's "Phishing
Websites" dataset.

### 3. Model (`src/train_model.py`)

A `RandomForestClassifier` (scikit-learn) trained on an 80/20 stratified
split. The script also supports `LogisticRegression` — change
`model_type="random_forest"` to `model_type="logistic"` in `main()`. Outputs:

- `model/phishing_model.pkl` — the trained model (via `joblib`)
- `model/feature_columns.pkl` — exact feature column order
- `model/metrics.json` — accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix, and feature importances

Current test-set performance (1,000 held-out rows): **~95% accuracy**,
**~98% precision**, **~91% recall** on the phishing class, **~0.98 ROC-AUC**.

### 4. App (`app.py`)

A single-file Streamlit app, styled as a monochrome "forensic scan report"
rather than a generic colorful dashboard:

- **Single Scan** — paste a URL, get a stamped verdict, a risk-confidence
  bar, and a full signal ledger (which features fired and why).
- **Batch Scan** — upload a CSV of URLs (or paste a list) and get a scored,
  downloadable results table.
- **Model Insights** — accuracy/precision/recall/F1/ROC-AUC, a grayscale
  confusion matrix, and a feature-importance chart.
- **About the Project** — this write-up, in-app.

## Limitations & disclaimer

This is an **educational demo**, not a production security tool:

- It judges URL *structure* only — it never visits the page, checks TLS
  certificate details, WHOIS domain age, or live threat-intel/blocklists.
- The training data is synthetic, not real captured traffic, so it will not
  perfectly reflect current real-world phishing campaigns.
- Both false positives (flagging a legitimate link) and false negatives
  (missing a real phishing link) are expected. Never treat a "looks
  legitimate" verdict as a guarantee, and always verify sensitive login
  pages through an independent channel.

## Extending this project

- Swap in a real-world dataset (see above) for production-grade accuracy.
- Add live signals: domain age (WHOIS), SSL certificate issuer/age, page
  content analysis, or a call to a threat-intel API.
- Try other models: XGBoost, gradient boosting, or a small neural net.
- Add a browser extension front end instead of/alongside Streamlit.
