# app.py  -- Data Storyteller Home

import streamlit as st
import pandas as pd
from ui.sidebar import render_sidebar


# ------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="📊 Data Storyteller",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
SAMPLE_DF = pd.DataFrame({
    "name": ["Honda City", "Maruti Swift", "Hyundai i20", "Toyota Corolla", "Ford Figo"],
    "company": ["Honda", "Maruti", "Hyundai", "Toyota", "Ford"],
    "year": [2018, 2016, 2019, 2015, 2017],
    "Price": ["₹550000", "₹380000", "₹610000", "₹720000", "₹290000"],
    "kms_driven": ["45,000", "60,500", "32,000", "80,000", "55,250"],
    "fuel_type": ["Petrol", "Petrol", "Diesel", "Petrol", "Diesel"],
})

sidebar_result = render_sidebar(
    app_pages=[
        {"path": "app.py", "label": "Home", "icon": "🏠"},
        {"path": "pages/1_Visual_Insights.py", "label": "Visual Insights", "icon": "📊"},
    ],
    show_file_uploader=False,  # we upload in main body
    sample_data=SAMPLE_DF,
    show_export=True,
    show_env_badge=True,
)


# ------------------------------------------------------------------
# Safe Imports of Internal Utils
# ------------------------------------------------------------------
try:
    from utils.analyzer import analyze_data
    from utils.summarizer import generate_summary
    from utils.pdf_generator import create_pdf  # returns bytes or BytesIO
except Exception as e:
    st.error(f"⚠️ Internal module import failed: {e}")
    st.stop()

# ------------------------------------------------------------------
# Session Defaults
# ------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
if "tone" not in st.session_state:
    st.session_state["tone"] = "formal"

# ------------------------------------------------------------------
# Theme Vars
# ------------------------------------------------------------------
THEME = st.session_state["theme"]
ACCENT_COLOR = "#3a7bd5" if THEME == "light" else "#00d2ff"
BG_CARD = "#f9f9f9" if THEME == "light" else "#1E1E1E"
TEXT_CARD = "#000" if THEME == "light" else "#fff"
BG_MAIN = "#ffffff" if THEME == "light" else "#121212"

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: "Segoe UI", Roboto, Arial, sans-serif !important;
        background-color: {BG_MAIN} !important;
        color: {TEXT_CARD} !important;
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #001f3f 0%, #0074D9 100%);
        color: white;
    }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: black !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
    }}
    /* Hero */
    .ds-hero {{
        padding: 1.5rem 2rem;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(0,122,255,0.15), rgba(0,255,200,0.15));
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        color: {TEXT_CARD};
    }}
    /* Cards / metrics */
    .suggestion-card {{
        border: 1px solid #ccc;
        padding: 12px;
        border-radius: 8px;
        background: {BG_CARD};
        color: {TEXT_CARD};
    }}
    div[data-testid="stMetric"] {{
        background: {BG_CARD};
        padding: 0.75rem;
        border-radius: 0.5rem;
        color: {TEXT_CARD};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# Hero Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="ds-hero">
        <h1>📊 Data Storyteller</h1>
        <p>Upload a CSV, explore quick stats, and generate narrative insights. 
        Dive into rich charts on the <strong>Visual Insights</strong> page (see sidebar).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# File Upload
# ------------------------------------------------------------------
st.markdown("### 📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])


# ------------------------------------------------------------------
# CSV Loader (cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file):
    try:
        return pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="ISO-8859-1")

# ------------------------------------------------------------------
# Load / Persist Data
# ------------------------------------------------------------------
df = None
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.session_state["data"] = df
        st.session_state["data_source_name"] = uploaded_file.name
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
elif "data" in st.session_state:
    df = st.session_state["data"]

# ------------------------------------------------------------------
# Cached Wrappers for Expensive Ops
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_analyze(df_in: pd.DataFrame):
    return analyze_data(df_in)

@st.cache_data(show_spinner=False)
def cached_summary(analysis_dict, tone: str):
    return generate_summary(analysis_dict, tone)

# ------------------------------------------------------------------
# Main Page Logic
# ------------------------------------------------------------------
if df is not None:
    st.subheader("🔍 Data Preview")
    max_rows = min(len(df), 1000)
    default_preview = max_rows if max_rows < 5 else 5
    num_rows = st.slider("Rows to preview", 1, max_rows, default_preview)
    st.dataframe(df.head(num_rows), use_container_width=True)
    st.markdown(f"✅ **{len(df):,} rows** | **{len(df.columns):,} columns**")

    # Analysis
    analysis = cached_analyze(df)

    # Metrics
    st.subheader("📈 Quick Analysis")
    shape = analysis.get("shape", {})
    rows = shape.get("rows", 0)
    cols = shape.get("columns", 0)
    column_types = analysis.get("column_types", {})

    col1, col2 = st.columns(2)
    col1.metric(label="Rows", value=f"{rows:,}")
    col2.metric(label="Columns", value=f"{cols:,}")

    st.markdown("### 🧾 Column Types")
    for colname, dtype in column_types.items():
        st.markdown(f"- **{colname}** → `{dtype}`")

    # Developer View
    with st.expander("🔬 Show raw analysis dictionary"):
        st.json(analysis)

    # Narrative Summary
    tone = st.session_state.get("tone", "formal")
    summary = cached_summary(analysis, tone)
    st.subheader("📝 Narrative Summary")
    st.markdown(summary)

    # Downloads
    st.download_button("⬇️ Download Summary (TXT)", summary, "data_summary.txt", "text/plain")
    try:
        pdf_buffer = create_pdf(summary)
        st.download_button("⬇️ Download Summary (PDF)", pdf_buffer, "data_summary.pdf", "application/pdf")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

else:
    st.info("👆 Upload a CSV file (or load sample data from the sidebar) to begin.")

st.markdown("---")
st.caption("Built with ❤️ by hericodes ")
