"""
app.py
------
Streamlit front end for the Phishing URL Detector.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from features import extract_features, FEATURE_NAMES, SUSPICIOUS_WORDS  # noqa: E402

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "model", "phishing_model.pkl")
COLS_PATH = os.path.join(ROOT, "model", "feature_columns.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")

st.set_page_config(
    page_title="URL Forensics — Phishing Detector",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# STYLE — monochrome "forensic scan report" theme
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --ink-000: #050505;
    --ink-100: #101010;
    --ink-200: #1B1B1B;
    --ink-300: #2A2A2A;
    --ink-400: #4A4A4A;
    --ink-500: #7A7A7A;
    --ink-600: #ABABAB;
    --ink-700: #D6D6D6;
    --ink-800: #F0F0F0;
    --paper:   #FAFAF8;
}

html, body, [class*="css"]  {
    font-family: 'JetBrains Mono', monospace;
}

.stApp {
    background: var(--ink-000);
    color: var(--ink-700);
}

h1, h2, h3, .display-font {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Kill Streamlit chrome we don't want */
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2.2rem; max-width: 1100px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ink-100);
    border-right: 1px solid var(--ink-300);
}
section[data-testid="stSidebar"] * { color: var(--ink-700) !important; }

/* Case-file header block */
.case-header {
    border: 1px solid var(--ink-300);
    border-left: 4px solid var(--ink-800);
    padding: 22px 26px;
    margin-bottom: 28px;
    background: linear-gradient(180deg, var(--ink-100), var(--ink-000));
}
.case-header .kicker {
    font-size: 0.72rem;
    color: var(--ink-500);
    letter-spacing: 0.14em;
    margin-bottom: 6px;
}
.case-header h1 {
    color: var(--ink-800);
    font-size: 2.0rem;
    margin: 0 0 6px 0;
}
.case-header p {
    color: var(--ink-500);
    margin: 0;
    font-size: 0.92rem;
    max-width: 640px;
    line-height: 1.5;
}

/* Section dividers */
.hairline {
    border: none;
    border-top: 1px solid var(--ink-300);
    margin: 26px 0;
}
.section-label {
    font-size: 0.72rem;
    color: var(--ink-500);
    letter-spacing: 0.1em;
    margin-bottom: 10px;
    display: block;
}

/* Verdict stamp */
.stamp-wrap { display: flex; justify-content: center; margin: 18px 0 6px 0; }
.stamp {
    display: inline-block;
    padding: 18px 40px;
    border: 3px double var(--stamp-color, var(--ink-800));
    color: var(--stamp-color, var(--ink-800));
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    letter-spacing: 0.06em;
    transform: rotate(-2.5deg);
    border-radius: 2px;
    text-align: center;
}
.stamp .sub {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    margin-top: 6px;
    opacity: 0.85;
}

/* Confidence bar */
.conf-row { display: flex; align-items: center; gap: 14px; margin: 18px 0; }
.conf-track {
    flex: 1;
    height: 10px;
    background: var(--ink-200);
    border: 1px solid var(--ink-300);
    position: relative;
    overflow: hidden;
}
.conf-fill { height: 100%; background: var(--ink-700); }
.conf-label { font-size: 0.85rem; color: var(--ink-600); min-width: 92px; text-align: right; }

/* Feature ledger table */
.ledger { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
.ledger th {
    text-align: left;
    color: var(--ink-500);
    font-weight: 500;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    padding: 8px 10px;
    border-bottom: 1px solid var(--ink-300);
}
.ledger td {
    padding: 9px 10px;
    border-bottom: 1px solid var(--ink-200);
    color: var(--ink-700);
}
.ledger tr:hover td { background: var(--ink-100); }
.flag-hit { color: var(--ink-800); font-weight: 600; }
.flag-clear { color: var(--ink-500); }

/* Metric cards */
.metric-card {
    border: 1px solid var(--ink-300);
    padding: 16px 18px;
    text-align: left;
    background: var(--ink-100);
}
.metric-card .val { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; color: var(--ink-800); font-weight: 700; }
.metric-card .lbl { font-size: 0.72rem; color: var(--ink-500); letter-spacing: 0.08em; margin-top: 4px; }

/* Buttons */
.stButton > button {
    background: var(--ink-800);
    color: var(--ink-000);
    border: 1px solid var(--ink-800);
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 0.55rem 1.4rem;
}
.stButton > button:hover {
    background: var(--ink-000);
    color: var(--ink-800);
    border: 1px solid var(--ink-600);
}

/* Text input */
.stTextInput input, .stTextArea textarea {
    background: var(--ink-100) !important;
    color: var(--ink-800) !important;
    border: 1px solid var(--ink-300) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.footnote { color: var(--ink-500); font-size: 0.78rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# DATA / MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
    return model, cols


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


model, feature_columns = load_model()
metrics = load_metrics()

FEATURE_LABELS = {
    "url_length": "URL length",
    "domain_length": "Domain length",
    "path_length": "Path length",
    "has_ip": "Host is a raw IP address",
    "has_at_symbol": "Contains '@' symbol",
    "has_https": "Uses HTTPS",
    "num_dots": "Number of dots",
    "num_hyphens": "Number of hyphens",
    "num_underscores": "Number of underscores",
    "num_digits": "Number of digits",
    "num_subdomains": "Number of subdomains",
    "has_suspicious_word": "Contains suspicious keyword",
    "has_shortener": "Uses a URL shortener",
    "has_suspicious_tld": "Suspicious top-level domain",
    "has_double_slash_redirect": "Double-slash redirect pattern",
    "num_query_params": "Number of query parameters",
    "has_port": "Explicit port in URL",
    "digit_letter_ratio": "Digit-to-letter ratio",
}

BOOLEAN_FEATURES = {
    "has_ip", "has_at_symbol", "has_https", "has_suspicious_word",
    "has_shortener", "has_suspicious_tld", "has_double_slash_redirect",
    "has_port",
}
# For these, a value of 1 is a RISK signal, except has_https where 1 is GOOD.
RISK_WHEN_ONE = BOOLEAN_FEATURES - {"has_https"}


def predict_url(url: str):
    feats = extract_features(url)
    X = pd.DataFrame([feats])[feature_columns]
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    return feats, pred, proba


def render_stamp(pred, proba_phish):
    if pred == 1:
        label, sub = "PHISHING DETECTED", f"confidence {proba_phish*100:.1f}%"
        color = "#F0F0F0"
    else:
        label, sub = "LOOKS LEGITIMATE", f"confidence {(1-proba_phish)*100:.1f}%"
        color = "#ABABAB"
    st.markdown(f"""
    <div class="stamp-wrap">
        <div class="stamp" style="--stamp-color:{color}">
            {label}
            <span class="sub">{sub}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence(proba_phish):
    pct = proba_phish * 100
    st.markdown(f"""
    <div class="conf-row">
        <div class="conf-track"><div class="conf-fill" style="width:{pct:.1f}%"></div></div>
        <div class="conf-label">{pct:.1f}% risk</div>
    </div>
    """, unsafe_allow_html=True)


def render_ledger(feats: dict):
    rows = []
    for name in feature_columns:
        val = feats[name]
        label = FEATURE_LABELS.get(name, name)
        if name in BOOLEAN_FEATURES:
            is_hit = (val == 1) if name in RISK_WHEN_ONE else (val == 0)
            display_val = "Yes" if val == 1 else "No"
            flag = "⚑ flagged" if is_hit else "clear"
            css = "flag-hit" if is_hit else "flag-clear"
        else:
            display_val = str(val)
            flag = ""
            css = "flag-clear"
        rows.append(f"<tr><td>{label}</td><td>{display_val}</td>"
                     f"<td class='{css}'>{flag}</td></tr>")

    st.markdown(f"""
    <table class="ledger">
        <tr><th>Signal</th><th>Value</th><th>Assessment</th></tr>
        {''.join(rows)}
    </table>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SIDEBAR NAV
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='display-font' style='font-size:1.25rem;color:#F0F0F0;font-weight:700;'>◼ URL FORENSICS</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#7A7A7A;font-size:0.78rem;margin-bottom:20px;'>Phishing URL Detector</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Single Scan", "Batch Scan", "Model Insights", "About the Project"],
        label_visibility="collapsed",
    )
    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    if metrics:
        st.markdown(f"""
        <div class="footnote">
        MODEL: RandomForest<br>
        TEST ACCURACY: {metrics['accuracy']*100:.1f}%<br>
        TRAINED ON: {metrics['n_train']+metrics['n_test']} samples
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='footnote'>Model not trained yet.<br>Run: python src/train_model.py</div>", unsafe_allow_html=True)


if model is None:
    st.error("No trained model found. Run `python src/train_model.py` from the project root first, "
             "then restart this app.")
    st.stop()


# --------------------------------------------------------------------------
# PAGE: SINGLE SCAN
# --------------------------------------------------------------------------
if page == "Single Scan":
    st.markdown("""
    <div class="case-header">
        <div class="kicker">CASE FILE · URL RISK ASSESSMENT</div>
        <h1>Inspect a link before you click it</h1>
        <p>Paste any URL below. The model breaks it down into structural signals —
        length, symbols, domain tricks, keywords — the same cues a trained eye
        looks for, and scores the likelihood it's a phishing attempt.</p>
    </div>
    """, unsafe_allow_html=True)

    example_col, _ = st.columns([3, 2])
    with example_col:
        examples = {
            "— pick an example —": "",
            "Likely legitimate": "https://www.github.com/settings/profile",
            "Likely phishing (IP host)": "http://192.168.14.9/paypal/verify.php",
            "Likely phishing (typosquat)": "https://paypa1-secure-login.tk/account/confirm",
        }
        choice = st.selectbox("Try an example", list(examples.keys()), label_visibility="collapsed")

    default_val = examples.get(choice, "")
    url_input = st.text_input("URL to analyze", value=default_val, placeholder="https://example.com/login")

    analyze = st.button("Analyze URL")

    if analyze and url_input.strip():
        feats, pred, proba = predict_url(url_input.strip())
        proba_phish = proba[1]

        st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
        render_stamp(pred, proba_phish)
        render_confidence(proba_phish)

        st.markdown("<span class='section-label'>SIGNAL LEDGER</span>", unsafe_allow_html=True)
        render_ledger(feats)

        st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>NOTE</span>", unsafe_allow_html=True)
        st.markdown(
            "<div class='footnote'>This is a heuristic/ML estimate based on URL structure only — "
            "it does not visit the page or check live blocklists. Treat a phishing flag as a reason "
            "to slow down, not a certainty, and verify through an independent channel before entering "
            "credentials.</div>", unsafe_allow_html=True
        )
    elif analyze:
        st.warning("Enter a URL first.")


# --------------------------------------------------------------------------
# PAGE: BATCH SCAN
# --------------------------------------------------------------------------
elif page == "Batch Scan":
    st.markdown("""
    <div class="case-header">
        <div class="kicker">CASE FILE · BULK INTAKE</div>
        <h1>Scan a list of URLs at once</h1>
        <p>Upload a CSV with a <code>url</code> column, or paste one URL per line.
        Every row is scored independently using the same model as the single-scan view.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Upload CSV", "Paste URLs"])
    urls = []

    with tab1:
        uploaded = st.file_uploader("CSV with a 'url' column", type=["csv"])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                url_col = "url" if "url" in df_in.columns else df_in.columns[0]
                urls = df_in[url_col].dropna().astype(str).tolist()
                st.caption(f"Loaded {len(urls)} URLs from column '{url_col}'.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    with tab2:
        pasted = st.text_area("One URL per line", height=160,
                               placeholder="https://example.com\nhttp://192.168.1.1/login")
        if pasted.strip():
            urls = [u.strip() for u in pasted.splitlines() if u.strip()]

    if urls:
        run = st.button(f"Scan {len(urls)} URLs")
        if run:
            records = []
            for u in urls:
                feats, pred, proba = predict_url(u)
                records.append({
                    "url": u,
                    "verdict": "Phishing" if pred == 1 else "Legitimate",
                    "risk_score_%": round(proba[1] * 100, 1),
                })
            result_df = pd.DataFrame(records).sort_values("risk_score_%", ascending=False)

            st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
            n_phish = (result_df["verdict"] == "Phishing").sum()
            c1, c2, c3 = st.columns(3)
            for col, val, lbl in zip(
                [c1, c2, c3],
                [len(result_df), n_phish, len(result_df) - n_phish],
                ["TOTAL SCANNED", "FLAGGED PHISHING", "LOOKS CLEAN"],
            ):
                col.markdown(f"<div class='metric-card'><div class='val'>{val}</div>"
                              f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv_bytes,
                                file_name=f"phishing_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv")


# --------------------------------------------------------------------------
# PAGE: MODEL INSIGHTS
# --------------------------------------------------------------------------
elif page == "Model Insights":
    st.markdown("""
    <div class="case-header">
        <div class="kicker">CASE FILE · MODEL PERFORMANCE</div>
        <h1>What's under the hood</h1>
        <p>Evaluation metrics from a held-out 20% test split, and which structural
        signals the model leans on most heavily.</p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.warning("No metrics found. Run `python src/train_model.py` first.")
    else:
        cols = st.columns(5)
        for col, (lbl, key) in zip(cols, [
            ("ACCURACY", "accuracy"), ("PRECISION", "precision"),
            ("RECALL", "recall"), ("F1 SCORE", "f1_score"), ("ROC AUC", "roc_auc"),
        ]):
            col.markdown(f"<div class='metric-card'><div class='val'>{metrics[key]*100:.1f}%</div>"
                          f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        left, right = st.columns(2)

        with left:
            st.markdown("<span class='section-label'>CONFUSION MATRIX</span>", unsafe_allow_html=True)
            cm = np.array(metrics["confusion_matrix"])

            mpl.rcParams.update({
                "text.color": "#D6D6D6", "axes.labelcolor": "#D6D6D6",
                "xtick.color": "#ABABAB", "ytick.color": "#ABABAB",
            })
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            fig.patch.set_facecolor("#0A0A0A")
            ax.set_facecolor("#0A0A0A")
            im = ax.imshow(cm, cmap="Greys")
            labels = ["Legitimate", "Phishing"]
            ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
            ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    txt_color = "#0A0A0A" if val > cm.max() / 2 else "#EDEDED"
                    ax.text(j, i, str(val), ha="center", va="center", color=txt_color, fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)

        with right:
            st.markdown("<span class='section-label'>TOP FEATURE IMPORTANCE</span>", unsafe_allow_html=True)
            if "feature_importances" in metrics:
                imp_df = pd.DataFrame(metrics["feature_importances"]).head(9)
                imp_df["label"] = imp_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
                imp_df = imp_df.sort_values("importance")

                fig2, ax2 = plt.subplots(figsize=(4.6, 3.6))
                fig2.patch.set_facecolor("#0A0A0A")
                ax2.set_facecolor("#0A0A0A")
                ax2.barh(imp_df["label"], imp_df["importance"], color="#ABABAB", edgecolor="#0A0A0A")
                ax2.tick_params(labelsize=8)
                ax2.set_xlabel("Importance")
                for spine in ["top", "right"]:
                    ax2.spines[spine].set_visible(False)
                for spine in ["left", "bottom"]:
                    ax2.spines[spine].set_color("#4A4A4A")
                fig2.tight_layout()
                st.pyplot(fig2)

        st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
        st.markdown(f"<div class='footnote'>Trained on {metrics['n_train']} samples · "
                     f"evaluated on {metrics['n_test']} held-out samples · model type: "
                     f"{metrics['model_type'].replace('_',' ').title()}.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE: ABOUT
# --------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="case-header">
        <div class="kicker">CASE FILE · PROJECT BRIEF</div>
        <h1>About this project</h1>
        <p>A self-contained, explainable machine-learning pipeline for
        flagging suspicious URLs from structure alone.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<span class='section-label'>PIPELINE</span>

**1. Feature extraction** (`src/features.py`) — every URL is converted into 18
numeric signals: length, dot/hyphen/digit counts, whether it's a raw IP,
whether it uses `@`, HTTPS presence, subdomain count, suspicious keywords
(login, verify, secure...), known URL shorteners, risky TLDs (.tk, .xyz...),
double-slash redirect tricks, query parameter count, and more.

**2. Dataset** (`data/generate_dataset.py`) — since this environment has no
internet access to pull a live phishing feed, the dataset is a **5,000-row
synthetic set** built from rules that mirror real phishing patterns (IP
hosts, typosquats, shorteners, sketchy TLDs) mixed with realistic legitimate
traffic, plus injected noise and edge cases so the classes aren't perfectly
separable — closer to what a real dataset looks like. Swap in a real dataset
(e.g. Kaggle's Phishing Website Detector or UCI's Phishing Websites dataset)
by dropping a CSV with `url` and `label` columns into `data/` and re-running.

**3. Model** (`src/train_model.py`) — a `RandomForestClassifier` (scikit-learn)
trained on an 80/20 split, saved to `model/phishing_model.pkl` with
`joblib`. Swap in `LogisticRegression` by passing `model_type="logistic"`.

**4. App** (`app.py`) — this Streamlit front end: single-URL scans, batch CSV
scans, and a model insights dashboard, all reading the same saved model.
    """)

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>SIGNALS USED</span>", unsafe_allow_html=True)
    feat_rows = "".join(
        f"<tr><td>{FEATURE_LABELS.get(f,f)}</td><td class='flag-clear'>{f}</td></tr>"
        for f in FEATURE_NAMES
    )
    st.markdown(f"""
    <table class="ledger">
        <tr><th>What it measures</th><th>Column name</th></tr>
        {feat_rows}
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footnote'><b>Disclaimer:</b> This is an educational demo, not a production "
        "security tool. It scores structural patterns only — it doesn't check page content, "
        "certificates, WHOIS age, or live threat-intel feeds — and both false positives and false "
        "negatives are expected. Don't rely on it as your only line of defense.</div>",
        unsafe_allow_html=True,
    )
