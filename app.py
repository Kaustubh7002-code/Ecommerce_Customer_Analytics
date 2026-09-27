from pathlib import Path
import re
import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="InsightIQ | Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .block-container { padding-top: 3.25rem !important; }
    .back-home-row { margin-top: 0.25rem; margin-bottom: 1.25rem; }
    .hero { margin-top: 0 !important; }
    

.stApp {
    background: #0b0f19;
    color: white;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263244;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
}

.subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #111827,
        #172554,
        #1e1b4b
    );
    border: 1px solid #293548;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 30px;
    font-weight: 750;
}

.hero-text {
    color: #cbd5e1;
    font-size: 16px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #263244;
}

.small-text {
    color: #9ca3af;
}


.home-nav { display:flex; align-items:center; justify-content:space-between; padding:12px 4px 24px 4px; }
.home-brand { font-size:26px; font-weight:800; letter-spacing:-0.5px; }
.home-brand span { color:#a5b4fc; }
.home-account { color:#cbd5e1; font-size:14px; }
.hero-home { padding:64px 42px 58px 42px; border-radius:26px; background:linear-gradient(135deg,#111827 0%,#172554 52%,#312e81 100%); border:1px solid #34415a; box-shadow:0 20px 60px rgba(0,0,0,0.28); margin-bottom:24px; }
.hero-badge { display:inline-block; padding:7px 14px; border-radius:999px; background:rgba(165,180,252,0.12); border:1px solid rgba(165,180,252,0.25); color:#c7d2fe; font-size:13px; font-weight:700; margin-bottom:18px; }
.hero-home-title { font-size:64px; line-height:1.05; font-weight:900; letter-spacing:-2px; margin-bottom:14px; }
.hero-home-subtitle { font-size:25px; font-weight:650; color:#e2e8f0; margin-bottom:16px; }
.hero-home-text { max-width:850px; margin:0 auto; color:#cbd5e1; font-size:18px; line-height:1.65; }
.feature-card { min-height:170px; padding:22px; border-radius:18px; background:#111827; border:1px solid #263244; }
.feature-card h3 { margin-top:0; margin-bottom:10px; }
.feature-card p { color:#aeb8c8; line-height:1.55; }
button[kind="primary"] { min-height:64px !important; font-size:21px !important; font-weight:800 !important; letter-spacing:0.4px !important; border-radius:14px !important; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA INTELLIGENCE / AUTOMATIC SCHEMA INFERENCE
# =========================================================

def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _date_parse_ratio(series):
    if len(series) == 0:
        return 0.0
    sample = series.dropna().astype(str).head(5000)
    if sample.empty:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _numeric_ratio(series):
    if len(series) == 0:
        return 0.0
    values = _safe_numeric(series)
    return float(values.notna().mean())


def _integer_ratio(series):
    values = _safe_numeric(series).dropna()
    if values.empty:
        return 0.0
    return float((np.isclose(values % 1, 0)).mean())


def _uniqueness_ratio(series):
    if len(series) == 0:
        return 1.0
    return float(series.nunique(dropna=True) / max(series.notna().sum(), 1))


def _repetition_score(series):
    ratio = _uniqueness_ratio(series)
    # Customer-like columns normally contain repeated entities.
    if ratio <= 0.02:
        return 1.0
    if ratio <= 0.10:
        return 0.95
    if ratio <= 0.30:
        return 0.80
    if ratio <= 0.60:
        return 0.55
    if ratio <= 0.85:
        return 0.25
    return 0.0


def _identifier_score(series):
    ratio = _uniqueness_ratio(series)
    if ratio >= 0.995:
        return 1.0
    if ratio >= 0.95:
        return 0.85
    if ratio >= 0.80:
        return 0.55
    return 0.10


def _amount_score(series):
    values = _safe_numeric(series).dropna()
    if values.empty:
        return 0.0
    positive_ratio = float((values >= 0).mean())
    if positive_ratio < 0.90:
        return 0.0
    nonzero = values[values > 0]
    if nonzero.empty:
        return 0.0
    # Amount-like columns usually have many distinct positive numeric values.
    distinct_ratio = nonzero.nunique() / max(len(nonzero), 1)
    decimal_ratio = 1.0 - _integer_ratio(nonzero)
    return min(1.0, 0.55 * distinct_ratio + 0.45 * max(decimal_ratio, 0.35))


def _quantity_score(series):
    """Score whether a numeric column behaves like transaction quantity.

    Age, year, rating and other numeric attributes are intentionally penalized.
    """
    numeric = _safe_numeric(series).dropna()
    if len(numeric) == 0:
        return 0.0

    integer_ratio = _integer_ratio(numeric)
    unique_ratio = _uniqueness_ratio(numeric)
    median = float(numeric.median())
    max_value = float(numeric.max())
    min_value = float(numeric.min())

    if min_value < 0 or max_value > 50:
        return 0.0

    if median > 15:
        size_score = 0.0
    elif median <= 5:
        size_score = 1.0
    elif median <= 10:
        size_score = 0.75
    else:
        size_score = 0.35

    cardinality_score = 1.0 if unique_ratio <= 0.05 else max(0.0, 1.0 - (unique_ratio - 0.05) / 0.25)
    return min(1.0, 0.60 * integer_ratio + 0.25 * size_score + 0.15 * cardinality_score)


def _text_like(series):
    if pd.api.types.is_numeric_dtype(series):
        return False
    return True


def _semantic_name_score(name, positive_words, negative_words=()):
    """Use column names only as a weak tie-breaker, never as a requirement."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split()
    positive = sum(1 for word in positive_words if word in normalized)
    negative = sum(1 for word in negative_words if word in normalized)
    return max(0.0, min(1.0, 0.25 * positive - 0.20 * negative))


def infer_transaction_schema(raw_df):
    """Infer an e-commerce transaction schema from data characteristics.

    Column names are deliberately NOT required. Names are not used as the
    source of truth; inference is based on data type, uniqueness, repetition,
    date parseability and numeric distributions.
    """
    work = raw_df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    profiles = []
    for col in work.columns:
        s = work[col]
        sample = s.dropna().head(5000)
        numeric_ratio = _numeric_ratio(sample)
        date_ratio = _date_parse_ratio(sample)
        unique_ratio = _uniqueness_ratio(sample)
        profiles.append({
            "column": col,
            "dtype": str(s.dtype),
            "numeric_ratio": numeric_ratio,
            "date_ratio": date_ratio,
            "unique_ratio": unique_ratio,
            "repetition": _repetition_score(sample),
            "identifier": _identifier_score(sample),
            "amount": _amount_score(sample),
            "quantity": _quantity_score(sample),
            "text": _text_like(s),
        })

    # ---------- DATE ----------
    date_candidates = sorted(
        profiles,
        key=lambda p: p["date_ratio"]
        + (0.15 if 0.05 <= p["unique_ratio"] <= 0.99 else 0),
        reverse=True,
    )
    date_p = date_candidates[0] if date_candidates and date_candidates[0]["date_ratio"] >= 0.75 else None

    # ---------- CUSTOMER ----------
    customer_candidates = [p for p in profiles if p is not date_p]
    def customer_cardinality_score(p):
        # Customer entities usually have more distinct values than country,
        # payment method, gender, category, etc., while still repeating.
        ratio = p["unique_ratio"]
        if 0.02 <= ratio <= 0.80:
            return 1.0
        if ratio < 0.02:
            return ratio / 0.02
        return max(0.0, 1.0 - (ratio - 0.80) / 0.20)

    customer_candidates = sorted(
        customer_candidates,
        key=lambda p: (
            0.55 * p["repetition"]
            + 0.35 * customer_cardinality_score(p)
            + 0.10 * (1.0 if p["text"] else 0)
        ),
        reverse=True,
    )
    customer_p = customer_candidates[0] if customer_candidates else None

    # If no useful repeated entity exists, a customer cannot be inferred safely.
    if customer_p and customer_p["repetition"] < 0.20:
        customer_p = None

    # ---------- ORDER / TRANSACTION ----------
    order_candidates = [p for p in profiles if p is not date_p and p is not customer_p]
    order_candidates = sorted(
        order_candidates,
        key=lambda p: (
            p["identifier"]
            + 0.15 * (1.0 if p["unique_ratio"] >= 0.80 else 0)
            + 0.10 * (1.0 if p["text"] else 0)
        ),
        reverse=True,
    )
    order_p = order_candidates[0] if order_candidates else None
    if order_p and order_p["identifier"] < 0.55:
        order_p = None

    # ---------- MONEY / REVENUE ----------
    amount_candidates = [p for p in profiles if p is not date_p and p is not customer_p and p is not order_p]
    amount_candidates = sorted(
        amount_candidates,
        key=lambda p: p["amount"],
        reverse=True,
    )
    amount_p = amount_candidates[0] if amount_candidates and amount_candidates[0]["amount"] >= 0.45 else None

    # ---------- QUANTITY ----------
    quantity_candidates = [p for p in profiles if p is not date_p and p is not customer_p and p is not order_p and p is not amount_p]
    quantity_candidates = sorted(
        quantity_candidates,
        key=lambda p: p["quantity"],
        reverse=True,
    )
    quantity_p = quantity_candidates[0] if quantity_candidates and quantity_candidates[0]["quantity"] >= 0.55 else None

    # ---------- PRODUCT / CATEGORY ----------
    product_candidates = [
        p for p in profiles
        if p is not date_p
        and p is not customer_p
        and p is not order_p
        and p is not amount_p
        and p is not quantity_p
        and p["text"]
    ]
    product_candidates = [
        p for p in product_candidates
        if 0.00005 <= p["unique_ratio"] <= 0.80
    ]
    product_candidates = sorted(
        product_candidates,
        key=lambda p: (
            0.45 * p["repetition"]
            + 0.25 * min(1.0, p["unique_ratio"] / 0.01)
            + 0.20 * _semantic_name_score(
                p["column"],
                ["product", "category", "item", "sku", "description", "brand"],
                ["country", "payment", "method", "age", "gender"]
            )
            + 0.10 * (1.0 if p["text"] else 0)
        ),
        reverse=True,
    )
    product_p = product_candidates[0] if product_candidates else None

    detected = {
        "customer": customer_p["column"] if customer_p else None,
        "order": order_p["column"] if order_p else None,
        "date": date_p["column"] if date_p else None,
        "amount": amount_p["column"] if amount_p else None,
        "quantity": quantity_p["column"] if quantity_p else None,
        "product": product_p["column"] if product_p else None,
    }

    confidence = {
        "customer": round(customer_p["repetition"] * 100, 1) if customer_p else 0.0,
        "order": round(order_p["identifier"] * 100, 1) if order_p else 0.0,
        "date": round(date_p["date_ratio"] * 100, 1) if date_p else 0.0,
        "amount": round(amount_p["amount"] * 100, 1) if amount_p else 0.0,
        "quantity": round(quantity_p["quantity"] * 100, 1) if quantity_p else 0.0,
        "product": round(product_p["repetition"] * 100, 1) if product_p else 0.0,
    }

    return detected, confidence, profiles


def standardize_transaction_data(raw_df, detected):
    """Convert the inferred schema into InsightIQ's internal schema."""
    df = raw_df.copy()

    customer_col = detected["customer"]
    order_col = detected["order"]
    date_col = detected["date"]
    amount_col = detected["amount"]
    quantity_col = detected["quantity"]
    product_col = detected["product"]

    if not customer_col or not date_col or not amount_col:
        raise ValueError(
            "The dataset does not contain enough information to perform "
            "customer transaction analytics. InsightIQ needs a repeated "
            "customer/buyer field, a date field, and a purchase amount field."
        )

    df["Customer_ID"] = df[customer_col].astype(str).str.strip()
    df["Order_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Revenue"] = _safe_numeric(df[amount_col])

    # Keep a Unit_Price field for compatibility with the rest of the dashboard.
    # The inferred amount is treated as transaction revenue; Unit_Price is derived
    # from it so the dashboard does not require a separate price column.

    # Quantity is optional. If absent, every transaction record represents one purchase.
    if quantity_col:
        quantity = _safe_numeric(df[quantity_col])
        # If a quantity candidate is actually a total amount duplicate, a 1 default
        # is safer than multiplying revenue incorrectly.
        df["Quantity"] = quantity
    else:
        df["Quantity"] = 1.0

    # Transaction/order ID is optional. A unique row ID is a safe fallback.
    if order_col:
        df["Order_ID"] = df[order_col].astype(str).str.strip()
    else:
        df["Order_ID"] = "TXN-" + (df.index.astype(str))

    if product_col:
        df["Product"] = df[product_col].astype(str).replace({"nan": "Unknown Product"})
    else:
        df["Product"] = "Unknown Product"

    # Remove empty/invalid identifiers and numerical values.
    df["Customer_ID"] = df["Customer_ID"].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    df["Order_ID"] = df["Order_ID"].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    df = df.dropna(subset=["Customer_ID", "Order_ID", "Order_Date", "Revenue"])
    df = df[df["Revenue"] > 0]

    # A missing quantity is already 1. For detected quantity, retain positive records.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1.0)
    df.loc[df["Quantity"] <= 0, "Quantity"] = 1.0
    df["Unit_Price"] = (
        df["Revenue"] / df["Quantity"]
    ).replace([np.inf, -np.inf], np.nan).fillna(df["Revenue"])

    return df


def schema_mapping_table(detected, confidence):
    rows = [
        ("Customer / Buyer", detected["customer"], confidence["customer"]),
        ("Transaction / Order", detected["order"], confidence["order"]),
        ("Transaction Date", detected["date"], confidence["date"]),
        ("Purchase Amount / Revenue", detected["amount"], confidence["amount"]),
        ("Quantity", detected["quantity"], confidence["quantity"]),
        ("Product / Category", detected["product"], confidence["product"]),
    ]
    return pd.DataFrame({
        "InsightIQ Field": [r[0] for r in rows],
        "Detected Column": [r[1] or "Not detected — adaptive fallback used" for r in rows],
        "Confidence": [f"{r[2]:.1f}%" if r[1] else "—" for r in rows],
    })


# =========================================================
# PAGE STATE
# =========================================================

# =========================================================

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


# =========================================================
# SIDEBAR
# =========================================================

page = "📊 Overview"

if st.session_state.analysis_started:

    with st.sidebar:

        st.markdown("## 🛒 InsightIQ")
        st.caption("E-commerce Intelligence Platform")

        st.markdown("---")

        page = st.radio(
            "NAVIGATION",
            [
                "📊 Overview",
                "👥 Customers",
                "💰 Revenue",
                "🎯 RFM Segmentation",
                "⚠️ Churn Analysis",
                "🤖 Churn Prediction",
                "📥 Reports"
            ],
            key="sidebar_page"
        )

        st.markdown("---")

        st.caption(
            "Python • Pandas • Scikit-learn • Streamlit"
        )


# =========================================================
# HOMEPAGE / ANALYSIS HEADER
# =========================================================

if st.session_state.analysis_started:

    st.markdown('<div class="back-home-row"></div>', unsafe_allow_html=True)
    back_col, spacer_col = st.columns([1.35, 6.65])
    with back_col:
        if st.button("🏠  Back to Home", type="secondary", key="back_to_home", use_container_width=True):
            st.session_state.analysis_started = False
            for key in ["analysis_df", "detected_schema", "schema_confidence", "schema_profiles", "file_signature"]:
                st.session_state.pop(key, None)
            st.session_state.pop("sidebar_page", None)
            st.rerun()

    st.markdown(
        '<div class="main-title">E-commerce Customer Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand customers • Analyze purchases • Predict churn'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD / DATA CACHE
# =========================================================

uploaded_file = None

if st.session_state.analysis_started:

    st.markdown(
        '<div class="section-title">📂 Upload Transaction Data</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload any e-commerce transaction CSV. InsightIQ profiles the actual "
        "data to infer the transaction structure instead of requiring fixed column names."
    )

    uploaded_file = st.file_uploader(
        "Choose a transaction CSV",
        type=["csv"],
        key="transaction_uploader"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_signature = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.get("file_signature") != file_signature:
            try:
                raw_df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

                if raw_df.empty or raw_df.shape[1] == 0:
                    raise ValueError("The uploaded CSV is empty.")

                raw_df.columns = [str(col).strip() for col in raw_df.columns]

                with st.spinner("🧠 Understanding your transaction data..."):
                    detected, confidence, profiles = infer_transaction_schema(raw_df)

                missing_core = [
                    field for field in ["customer", "date", "amount"]
                    if not detected[field]
                ]

                if missing_core:
                    st.error(
                        "InsightIQ could not confidently identify the core transaction fields."
                    )
                    st.write(
                        "The file needs enough information to infer a repeated customer/buyer "
                        "field, a transaction date, and a purchase amount. Column names are not "
                        "required; the system analyzes the values and data types."
                    )
                    st.dataframe(
                        pd.DataFrame(profiles)[
                            ["column", "dtype", "numeric_ratio", "date_ratio", "unique_ratio"]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )
                    st.stop()

                processed_df = standardize_transaction_data(raw_df, detected)

                if processed_df.empty:
                    raise ValueError(
                        "No usable transaction records remain after data validation. "
                        "Please upload a transaction dataset with valid dates, customers and purchase amounts."
                    )

                st.session_state["analysis_df"] = processed_df
                st.session_state["detected_schema"] = detected
                st.session_state["schema_confidence"] = confidence
                st.session_state["schema_profiles"] = profiles
                st.session_state["file_signature"] = file_signature
                st.session_state.pop("model_results", None)

            except UnicodeDecodeError:
                try:
                    raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", low_memory=False)
                    raw_df.columns = [str(col).strip() for col in raw_df.columns]
                    detected, confidence, profiles = infer_transaction_schema(raw_df)
                    processed_df = standardize_transaction_data(raw_df, detected)
                    if processed_df.empty:
                        raise ValueError("No usable transaction records remain after data validation.")
                    st.session_state["analysis_df"] = processed_df
                    st.session_state["detected_schema"] = detected
                    st.session_state["schema_confidence"] = confidence
                    st.session_state["schema_profiles"] = profiles
                    st.session_state["file_signature"] = file_signature
                    st.session_state.pop("model_results", None)
                except Exception as e:
                    st.error(f"Unable to process the uploaded CSV: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Unable to process the uploaded CSV: {e}")
                st.stop()

    if "analysis_df" in st.session_state:
        df = st.session_state["analysis_df"].copy()
        detected = st.session_state.get("detected_schema", {})
        confidence = st.session_state.get("schema_confidence", {})

        st.success(
            f"CSV understood successfully — {len(df):,} usable transaction records found."
        )

        with st.expander("🧠 InsightIQ's Automatic Data Understanding", expanded=False):
            st.dataframe(
                schema_mapping_table(detected, confidence),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "Detection uses data types, value patterns, uniqueness and repetition. "
                "Column names are not required."
            )

            if not detected.get("order"):
                st.info("No reliable transaction ID was detected. A unique transaction ID was generated for each record.")
            if not detected.get("quantity"):
                st.info("No reliable quantity field was detected. Each transaction record is treated as one purchase for item-count analysis.")
            if not detected.get("product"):
                st.info("No product/category field was detected. Product-specific analysis will use an Unknown Product fallback.")

# =========================================================
# NO FILE / HOMEPAGE
# =========================================================

if not st.session_state.analysis_started:

    st.markdown(
        """
        <div class="home-nav">
            <div class="home-brand">🧠 <span>InsightIQ</span></div>
            <div></div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="hero-home" style="text-align:center;">
            <div class="hero-badge">📈 DATA • ANALYTICS • MACHINE LEARNING</div>
            <div class="hero-home-title">🧠 InsightIQ</div>
            <div class="hero-home-subtitle">E-Commerce Customer Intelligence Platform</div>
            <div class="hero-home-text">
                Turn transaction data into meaningful customer insights.
                Understand purchasing behavior, analyze revenue, segment customers,
                and identify potential churn risk with analytics and machine learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cta1, cta2, cta3 = st.columns([1.8, 3.4, 1.8])
    with cta2:
        if st.button("🚀  START ANALYSIS", use_container_width=True, type="primary", key="start_analysis_home"):
            st.session_state.analysis_started = True
            st.rerun()

    st.markdown(
        '<div class="section-title">✨ Everything You Need in One Platform</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("📊", "Customer Analytics", "Understand customers, orders, spending and purchasing behavior."),
        ("💰", "Revenue Analysis", "Explore revenue trends, products, orders and average order value."),
        ("🎯", "RFM Segmentation", "Group customers using Recency, Frequency and Monetary value."),
        ("🤖", "Churn Prediction", "Use machine learning to estimate customer churn risk."),
    ]
    for col, (icon, title, desc) in zip((c1, c2, c3, c4), cards):
        with col:
            st.markdown(
                f'<div class="feature-card"><div style="font-size:30px;">{icon}</div><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">🔄 Simple Workflow</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "**1. Upload Data**  →  **2. Clean & Validate**  →  **3. Analyze**  →  "
        "**4. Segment**  →  **5. Predict Churn**  →  **6. Get Insights**"
    )

    st.markdown(
        '<div class="section-title">💡 About InsightIQ</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="home-about" style="background:#111827; border:1px solid #263244; border-radius:18px; padding:26px 30px; line-height:1.7; color:#cbd5e1;">
            <strong style="color:#ffffff; font-size:20px;">InsightIQ</strong> is an end-to-end
            <strong style="color:#ffffff;">E-Commerce Customer Intelligence Platform</strong> designed
            to help businesses understand their customers and make data-driven decisions.
            <br><br>
            It transforms raw transaction data into meaningful insights by analyzing
            <strong style="color:#ffffff;">customer behavior, purchasing patterns, revenue performance and customer value</strong>.
            Businesses can use <strong style="color:#ffffff;">RFM segmentation</strong> to identify valuable
            customer groups and <strong style="color:#ffffff;">machine-learning-based churn prediction</strong>
            to identify customers who may be at risk of leaving.
            <br><br>
            The platform is designed to work with a company's own transaction data, making it useful
            for exploring customer trends, identifying opportunities and supporting data-driven
            business strategies.
            <div style="margin-top:18px; padding-top:16px; border-top:1px solid #263244;">
                <strong style="color:#ffffff;">Upload → Analyze → Understand → Predict → Take Action</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

if "analysis_df" not in st.session_state:
    st.info("👆 Upload a transaction CSV above to begin the analysis.")
    st.stop()

# =========================================================
# DATA CLEANING
# =========================================================

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Unit_Price"] = pd.to_numeric(
    df["Unit_Price"],
    errors="coerce"
)

df["Revenue"] = pd.to_numeric(
    df["Revenue"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Customer_ID",
        "Order_ID",
        "Order_Date",
        "Quantity",
        "Revenue"
    ]
)

df = df[df["Quantity"] > 0]
df = df[df["Revenue"] > 0]

# Unit price is a derived compatibility field when the uploaded dataset
# supplied transaction-level revenue rather than a separate unit price.

df["Customer_ID"] = (
    df["Customer_ID"]
    .astype(str)
    .str.strip()
)

if df.empty:
    st.error("No usable transaction records remain after cleaning. Check your dates, quantities, prices, and customer/order IDs.")
    st.stop()


# =========================================================
# CUSTOMER ANALYTICS
# =========================================================

customer_data = df.groupby(
    "Customer_ID"
).agg(
    Total_Spend=("Revenue", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Items=("Quantity", "sum"),
    First_Purchase=("Order_Date", "min"),
    Last_Purchase=("Order_Date", "max")
).reset_index()

customer_data["Average_Order_Value"] = (
    customer_data["Total_Spend"] /
    customer_data["Total_Orders"]
)

analysis_date = df["Order_Date"].max() + pd.Timedelta(days=1)

customer_data["Recency"] = (
    analysis_date -
    customer_data["Last_Purchase"]
).dt.days


# =========================================================
# RFM
# =========================================================

rfm = customer_data[
    [
        "Customer_ID",
        "Recency",
        "Total_Orders",
        "Total_Spend"
    ]
].copy()

rfm = rfm.rename(
    columns={
        "Total_Orders": "Frequency",
        "Total_Spend": "Monetary"
    }
)


# RFM scores

try:

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"],
        5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop"
    )

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    )

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5]
    )

except:

    rfm["R_Score"] = 3
    rfm["F_Score"] = 3
    rfm["M_Score"] = 3


rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)


# =========================================================
# SEGMENTS
# =========================================================

def assign_segment(row):

    r = int(row["R_Score"])
    f = int(row["F_Score"])
    m = int(row["M_Score"])

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4:
        return "Loyal Customers"

    elif r >= 4 and f <= 2:
        return "New Customers"

    elif r <= 2 and f >= 3:
        return "At Risk"

    elif r <= 2 and f <= 2:
        return "Lost Customers"

    else:
        return "Potential Loyalists"


rfm["Segment"] = rfm.apply(
    assign_segment,
    axis=1
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "📊 Overview":

    st.markdown(
        '<div class="section-title">📊 Business Overview</div>',
        unsafe_allow_html=True
    )

    total_customers = df["Customer_ID"].nunique()

    total_orders = df["Order_ID"].nunique()

    total_revenue = df["Revenue"].sum()

    average_order_value = (
        total_revenue / total_orders if total_orders > 0 else 0
    )

    returning_customers = (
        customer_data["Total_Orders"] > 1
    ).sum()

    repeat_rate = (
        returning_customers /
        total_customers
    ) * 100


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "🛒 Total Orders",
            f"{total_orders:,}"
        )

    with col3:

        st.metric(
            "💰 Total Revenue",
            f"£{total_revenue:,.2f}"
        )

    with col4:

        st.metric(
            "📦 Average Order Value",
            f"£{average_order_value:,.2f}"
        )


    col5, col6 = st.columns(2)

    with col5:

        st.metric(
            "🔁 Repeat Customer Rate",
            f"{repeat_rate:.2f}%"
        )

    with col6:

        st.metric(
            "📅 Data Period",
            f"{df['Order_Date'].min().date()} → "
            f"{df['Order_Date'].max().date()}"
        )


    # Revenue chart

    st.markdown(
        '<div class="section-title">📈 Revenue Trend</div>',
        unsafe_allow_html=True
    )

    monthly_revenue = (
        df.set_index("Order_Date")
        .resample("ME")["Revenue"]
        .sum()
    )

    st.line_chart(
        monthly_revenue
    )


    # Segment chart

    st.markdown(
        '<div class="section-title">🎯 Customer Segments</div>',
        unsafe_allow_html=True
    )

    segment_counts = (
        rfm["Segment"]
        .value_counts()
    )

    st.bar_chart(
        segment_counts
    )


# =========================================================
# CUSTOMERS
# =========================================================

elif page == "👥 Customers":

    st.markdown(
        '<div class="section-title">👥 Customer Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers",
            f"{len(customer_data):,}"
        )

    with col2:

        st.metric(
            "Repeat Customers",
            f"{(customer_data['Total_Orders'] > 1).sum():,}"
        )

    with col3:

        st.metric(
            "Average Customer Spend",
            f"£{customer_data['Total_Spend'].mean():,.2f}"
        )


    st.markdown(
        '<div class="section-title">🏆 Highest Value Customers</div>',
        unsafe_allow_html=True
    )

    top_customers = (
        customer_data
        .sort_values(
            "Total_Spend",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top_customers[
            [
                "Customer_ID",
                "Total_Orders",
                "Total_Items",
                "Total_Spend",
                "Average_Order_Value",
                "Last_Purchase"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# REVENUE
# =========================================================

elif page == "💰 Revenue":

    st.markdown(
        '<div class="section-title">💰 Revenue Analytics</div>',
        unsafe_allow_html=True
    )

    total_revenue = df["Revenue"].sum()

    total_orders = df["Order_ID"].nunique()

    average_order_value = (
        total_revenue / total_orders if total_orders > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Revenue",
            f"£{total_revenue:,.2f}"
        )

    with col2:

        st.metric(
            "Total Orders",
            f"{total_orders:,}"
        )

    with col3:

        st.metric(
            "Average Order Value",
            f"£{average_order_value:,.2f}"
        )


    st.markdown(
        '<div class="section-title">📈 Monthly Revenue</div>',
        unsafe_allow_html=True
    )

    monthly = (
        df.set_index("Order_Date")
        .resample("ME")["Revenue"]
        .sum()
    )

    st.line_chart(monthly)


    st.markdown(
        '<div class="section-title">📦 Top Products</div>',
        unsafe_allow_html=True
    )

    product_sales = (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    st.bar_chart(product_sales)


# =========================================================
# RFM
# =========================================================

elif page == "🎯 RFM Segmentation":

    st.markdown(
        '<div class="section-title">🎯 RFM Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Customers are segmented using Recency, Frequency "
        "and Monetary value."
    )


    segment_counts = (
        rfm["Segment"]
        .value_counts()
    )

    st.bar_chart(
        segment_counts
    )


    st.markdown(
        '<div class="section-title">Customer Segments</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        rfm[
            [
                "Customer_ID",
                "Recency",
                "Frequency",
                "Monetary",
                "RFM_Score",
                "Segment"
            ]
        ].sort_values(
            "Monetary",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CHURN ANALYSIS
# =========================================================

elif page == "⚠️ Churn Analysis":

    st.markdown(
        '<div class="section-title">⚠️ Churn Analysis</div>',
        unsafe_allow_html=True
    )

    churn_threshold = st.number_input(
        "Days without purchase considered at-risk",
        min_value=1,
        value=90,
        step=1
    )

    customer_data["Churn_Status"] = np.where(
        customer_data["Recency"] > churn_threshold,
        "Churned / Inactive",
        "Active"
    )

    status_counts = (
        customer_data["Churn_Status"]
        .value_counts()
    )

    col1, col2, col3 = st.columns(3)

    active = (
        customer_data["Churn_Status"]
        == "Active"
    ).sum()

    churned = (
        customer_data["Churn_Status"]
        == "Churned / Inactive"
    ).sum()

    churn_rate = (
        churned /
        len(customer_data)
    ) * 100

    with col1:
        st.metric(
            "Active Customers",
            f"{active:,}"
        )

    with col2:
        st.metric(
            "Inactive / Churned",
            f"{churned:,}"
        )

    with col3:
        st.metric(
            "Inactivity Rate",
            f"{churn_rate:.2f}%"
        )


    st.bar_chart(status_counts)


    st.markdown(
        '<div class="section-title">'
        '🚨 Customers At Risk'
        '</div>',
        unsafe_allow_html=True
    )

    at_risk = (
        customer_data[
            customer_data["Recency"]
            > churn_threshold
        ]
        .sort_values(
            "Total_Spend",
            ascending=False
        )
    )

    st.dataframe(
        at_risk[
            [
                "Customer_ID",
                "Recency",
                "Total_Orders",
                "Total_Spend",
                "Average_Order_Value"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CHURN PREDICTION
# =========================================================

elif page == "🤖 Churn Prediction":

    st.title("🤖 Churn Prediction")
    st.write(
        "Train a customer churn model using the uploaded transaction data."
    )

    st.info(
        "The model is trained from the uploaded dataset instead of using "
        "a fixed pre-trained dataset model."
    )

    # ---------------------------------------------------------
    # CHURN WINDOW
    # ---------------------------------------------------------

    churn_window = st.number_input(
        "Churn observation window (days)",
        min_value=1,
        value=90,
        step=1
    )

    st.caption(
        f"A customer is considered churned if they make no purchase "
        f"during the {churn_window}-day observation period."
    )

    # ---------------------------------------------------------
    # BASIC DATA VALIDATION
    # ---------------------------------------------------------

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        st.stop()

    if "Order_Date" not in df.columns:
        st.error("Order date column is required for churn prediction.")
        st.stop()

    if "Customer_ID" not in df.columns:
        st.error("Customer ID column is required for churn prediction.")
        st.stop()

    # Make sure dates are valid
    churn_df = df.copy()

    churn_df["Order_Date"] = pd.to_datetime(
        churn_df["Order_Date"],
        errors="coerce"
    )

    churn_df = churn_df.dropna(
        subset=["Order_Date", "Customer_ID"]
    )

    if churn_df.empty:
        st.error("No valid transaction dates were found.")
        st.stop()

    # ---------------------------------------------------------
    # CHECK DATA PERIOD
    # ---------------------------------------------------------

    min_date = churn_df["Order_Date"].min()
    max_date = churn_df["Order_Date"].max()

    total_days = (max_date - min_date).days

    st.write(
        f"**Data period:** {min_date.date()} → {max_date.date()}"
    )

    st.write(
        f"**Available history:** {total_days} days"
    )

    minimum_required_days = churn_window * 2

    if total_days < minimum_required_days:
        st.error(
            f"Not enough historical data for churn prediction. "
            f"At least {minimum_required_days} days of transaction history "
            f"is recommended for a {churn_window}-day churn window."
        )
        st.stop()

    # ---------------------------------------------------------
    # CREATE HISTORICAL + FUTURE PERIOD
    # ---------------------------------------------------------

    prediction_cutoff = max_date - pd.Timedelta(
        days=churn_window
    )

    historical_df = churn_df[
        churn_df["Order_Date"] < prediction_cutoff
    ].copy()

    future_df = churn_df[
        churn_df["Order_Date"] >= prediction_cutoff
    ].copy()

    if historical_df.empty or future_df.empty:
        st.error(
            "Unable to create historical and future periods."
        )
        st.stop()

    # ---------------------------------------------------------
    # CREATE HISTORICAL CUSTOMER FEATURES
    # ---------------------------------------------------------

    feature_date = historical_df["Order_Date"].max() + pd.Timedelta(days=1)

    customer_features = (
        historical_df
        .groupby("Customer_ID")
        .agg(
            Recency=(
                "Order_Date",
                lambda x: (feature_date - x.max()).days
            ),
            Frequency=(
                "Order_ID",
                "nunique"
            ),
            Monetary=(
                "Revenue",
                "sum"
            ),
            Total_Items=(
                "Quantity",
                "sum"
            ),
            First_Purchase=(
                "Order_Date",
                "min"
            ),
            Last_Purchase=(
                "Order_Date",
                "max"
            )
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # ADD AVERAGE ORDER VALUE
    # ---------------------------------------------------------

    customer_features["Average_Order_Value"] = (
        customer_features["Monetary"]
        / customer_features["Frequency"].replace(0, np.nan)
    )

    customer_features["Average_Order_Value"] = (
        customer_features["Average_Order_Value"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # CUSTOMER LIFETIME
    # ---------------------------------------------------------

    customer_features["Customer_Lifetime_Days"] = (
        customer_features["Last_Purchase"]
        - customer_features["First_Purchase"]
    ).dt.days

    customer_features["Customer_Lifetime_Days"] = (
        customer_features["Customer_Lifetime_Days"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # CREATE FUTURE PURCHASE INFORMATION
    # ---------------------------------------------------------

    future_orders = (
        future_df
        .groupby("Customer_ID")["Order_ID"]
        .nunique()
        .reset_index()
    )

    future_orders.columns = [
        "Customer_ID",
        "Future_Orders"
    ]

    customer_features = customer_features.merge(
        future_orders,
        on="Customer_ID",
        how="left"
    )

    customer_features["Future_Orders"] = (
        customer_features["Future_Orders"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # CREATE CHURN LABEL
    # ---------------------------------------------------------

    customer_features["Churn"] = (
        customer_features["Future_Orders"] == 0
    ).astype(int)

    # ---------------------------------------------------------
    # CHECK CLASS BALANCE
    # ---------------------------------------------------------

    churn_count = customer_features["Churn"].sum()
    active_count = (
        customer_features["Churn"] == 0
    ).sum()

    total_customers = len(customer_features)

    if churn_count == 0 or active_count == 0:
        st.error(
            "The dataset does not contain both churned and active "
            "customers. Try a different churn window."
        )
        st.stop()

    # ---------------------------------------------------------
    # DISPLAY TRAINING INFORMATION
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Customers for Training",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churn_count:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{(churn_count / total_customers) * 100:.1f}%"
        )

    # ---------------------------------------------------------
    # MACHINE LEARNING FEATURES
    # ---------------------------------------------------------

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Items",
        "Average_Order_Value",
        "Customer_Lifetime_Days"
    ]

    X = customer_features[features].copy()
    y = customer_features["Churn"].copy()

    # Remove invalid values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_rows = X.notna().all(axis=1)

    X = X[valid_rows]
    y = y[valid_rows]

    if len(X) < 20:
        st.error(
            "Not enough customers to train a reliable churn model."
        )
        st.stop()

    if y.nunique() < 2:
        st.error(
            "The churn target contains only one class. "
            "Try changing the churn observation window."
        )
        st.stop()

    # ---------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # RANDOM FOREST MODEL
    # ---------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )
    except ValueError:
        roc_auc = 0

    st.subheader("📊 Model Performance")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with metric2:
        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with metric3:
        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with metric4:
        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    with metric5:
        st.metric(
            "ROC-AUC",
            f"{roc_auc:.2%}"
        )

    st.caption(
        "These metrics are calculated on a held-out test set from "
        "the uploaded dataset."
    )

    # ---------------------------------------------------------
    # CURRENT CUSTOMER DATA
    # ---------------------------------------------------------

    current_date = churn_df["Order_Date"].max() + pd.Timedelta(days=1)

    current_customer_data = (
        churn_df
        .groupby("Customer_ID")
        .agg(
            Recency=(
                "Order_Date",
                lambda x: (current_date - x.max()).days
            ),
            Frequency=(
                "Order_ID",
                "nunique"
            ),
            Monetary=(
                "Revenue",
                "sum"
            ),
            Total_Items=(
                "Quantity",
                "sum"
            ),
            First_Purchase=(
                "Order_Date",
                "min"
            ),
            Last_Purchase=(
                "Order_Date",
                "max"
            )
        )
        .reset_index()
    )

    current_customer_data["Average_Order_Value"] = (
        current_customer_data["Monetary"]
        / current_customer_data["Frequency"].replace(
            0,
            np.nan
        )
    )

    current_customer_data["Average_Order_Value"] = (
        current_customer_data["Average_Order_Value"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # CUSTOMER LIFETIME
    # ---------------------------------------------------------

    current_customer_data["Customer_Lifetime_Days"] = (
        current_customer_data["Last_Purchase"]
        - current_customer_data["First_Purchase"]
    ).dt.days

    current_customer_data["Customer_Lifetime_Days"] = (
        current_customer_data["Customer_Lifetime_Days"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ---------------------------------------------------------
    # PREPARE CURRENT CUSTOMER FEATURES
    # ---------------------------------------------------------

    prediction_features = current_customer_data[features].copy()

    prediction_features = prediction_features.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    # ---------------------------------------------------------
    # PREDICT CHURN PROBABILITY
    # ---------------------------------------------------------

    current_customer_data["Churn_Probability"] = (
        model.predict_proba(prediction_features)[:, 1]
    )

    # ---------------------------------------------------------
    # ASSIGN CUSTOMER RISK LEVEL
    # ---------------------------------------------------------

    current_customer_data["Risk_Level"] = pd.cut(
        current_customer_data["Churn_Probability"],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"]
    )

    # ---------------------------------------------------------
    # PREDICTION SUMMARY
    # ---------------------------------------------------------

    st.subheader("🎯 Current Customer Churn Risk")

    risk_counts = current_customer_data["Risk_Level"].value_counts().reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )

    risk1, risk2, risk3 = st.columns(3)

    with risk1:
        st.metric(
            "Low Risk",
            f"{risk_counts['Low']:,}"
        )

    with risk2:
        st.metric(
            "Medium Risk",
            f"{risk_counts['Medium']:,}"
        )

    with risk3:
        st.metric(
            "High Risk",
            f"{risk_counts['High']:,}"
        )

    # ---------------------------------------------------------
    # RISK DISTRIBUTION CHART
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        risk_counts.index.astype(str),
        risk_counts.values
    )

    ax.set_title("Customer Churn Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------
    # HIGH-RISK CUSTOMERS
    # ---------------------------------------------------------

    st.subheader("🚨 Customers at Highest Churn Risk")

    high_risk_customers = (
        current_customer_data[
            current_customer_data["Risk_Level"] == "High"
        ]
        .sort_values("Churn_Probability", ascending=False)
        .copy()
    )

    if high_risk_customers.empty:
        st.success("No customers are currently classified as High Risk.")
    else:
        high_risk_display = high_risk_customers[
            [
                "Customer_ID",
                "Recency",
                "Frequency",
                "Monetary",
                "Average_Order_Value",
                "Churn_Probability",
                "Risk_Level"
            ]
        ].copy()

        high_risk_display["Churn_Probability"] = (
            high_risk_display["Churn_Probability"] * 100
        ).round(2)

        high_risk_display["Monetary"] = (
            high_risk_display["Monetary"].round(2)
        )

        high_risk_display["Average_Order_Value"] = (
            high_risk_display["Average_Order_Value"].round(2)
        )

        high_risk_display = high_risk_display.rename(
            columns={
                "Churn_Probability": "Churn Probability (%)",
                "Average_Order_Value": "Average Order Value"
            }
        )

        st.dataframe(
            high_risk_display,
            use_container_width=True,
            hide_index=True
        )

    # ---------------------------------------------------------
    # ALL CUSTOMER PREDICTIONS
    # ---------------------------------------------------------

    st.subheader("👥 All Customer Predictions")

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

    filtered_predictions = current_customer_data[
        current_customer_data["Risk_Level"].astype(str).isin(risk_filter)
    ].copy()

    filtered_predictions = filtered_predictions.sort_values(
        "Churn_Probability",
        ascending=False
    )

    prediction_display = filtered_predictions[
        [
            "Customer_ID",
            "Recency",
            "Frequency",
            "Monetary",
            "Total_Items",
            "Average_Order_Value",
            "Customer_Lifetime_Days",
            "Churn_Probability",
            "Risk_Level"
        ]
    ].copy()

    prediction_display["Churn_Probability"] = (
        prediction_display["Churn_Probability"] * 100
    ).round(2)

    prediction_display["Monetary"] = (
        prediction_display["Monetary"].round(2)
    )

    prediction_display["Average_Order_Value"] = (
        prediction_display["Average_Order_Value"].round(2)
    )

    prediction_display = prediction_display.rename(
        columns={
            "Churn_Probability": "Churn Probability (%)",
            "Average_Order_Value": "Average Order Value",
            "Customer_Lifetime_Days": "Customer Lifetime (Days)"
        }
    )

    st.dataframe(
        prediction_display,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    st.subheader("🔎 What Influences Churn Prediction?")

    rf_model = model.named_steps["model"]

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1]
    )

    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Feature importance shows how much each input feature contributes "
        "to the Random Forest's decisions. It does not prove that a feature "
        "causes churn."
    )

    # ---------------------------------------------------------
    # DOWNLOAD PREDICTIONS
    # ---------------------------------------------------------

    st.subheader("📥 Download Churn Predictions")

    download_columns = [
        "Customer_ID",
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Items",
        "Average_Order_Value",
        "Customer_Lifetime_Days",
        "First_Purchase",
        "Last_Purchase",
        "Churn_Probability",
        "Risk_Level"
    ]

    download_df = current_customer_data[download_columns].copy()

    download_df["Churn_Probability"] = (
        download_df["Churn_Probability"] * 100
    ).round(2)

    csv_data = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Customer Churn Predictions CSV",
        data=csv_data,
        file_name="customer_churn_predictions.csv",
        mime="text/csv"
    )

# =========================================================
# REPORTS
# =========================================================

elif page == "📥 Reports":

    st.markdown(
        '<div class="section-title">📥 Download Reports</div>',
        unsafe_allow_html=True
    )

    customer_csv = customer_data.to_csv(
        index=False
    ).encode("utf-8")

    rfm_csv = rfm.to_csv(index=False).encode("utf-8")
    revenue_summary = pd.DataFrame({
        "Metric": ["Total Revenue", "Total Customers", "Total Orders", "Average Order Value", "Repeat Customer Rate (%)"],
        "Value": [
            df["Revenue"].sum(),
            df["Customer_ID"].nunique(),
            df["Order_ID"].nunique(),
            df["Revenue"].sum() / df["Order_ID"].nunique() if df["Order_ID"].nunique() else 0,
            ((customer_data["Total_Orders"] > 1).sum() / len(customer_data) * 100) if len(customer_data) else 0
        ]
    })
    revenue_csv = revenue_summary.to_csv(index=False).encode("utf-8")
    cleaned_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("⬇️ Download Customer Analytics", customer_csv, "customer_analytics.csv", "text/csv")
    st.download_button("⬇️ Download RFM Segmentation", rfm_csv, "rfm_segmentation.csv", "text/csv")
    st.download_button("⬇️ Download Revenue Summary", revenue_csv, "revenue_summary.csv", "text/csv")
    st.download_button("⬇️ Download Cleaned Transactions", cleaned_csv, "cleaned_transactions.csv", "text/csv")


    st.markdown(
        '<div class="section-title">'
        '📊 Current Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Records: {len(df):,}"
    )

    st.write(
        f"Customers: {df['Customer_ID'].nunique():,}"
    )

    st.write(
        f"Orders: {df['Order_ID'].nunique():,}"
    )

    st.write(
        f"Revenue: £{df['Revenue'].sum():,.2f}"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<br><br><center>"
    "<span style='color:#64748b'>"
    "InsightIQ • E-commerce Customer Analytics & Churn Prediction"
    "</span>"
    "</center>",
    unsafe_allow_html=True
)