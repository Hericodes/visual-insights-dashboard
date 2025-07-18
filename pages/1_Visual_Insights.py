import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Tuple, List, Optional

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Matplotlib (fallback export only)
import matplotlib.pyplot as plt  # noqa

# =============================================================================
# Page Config (must be first Streamlit call in file execution context)
# =============================================================================
st.set_page_config(page_title="📊 Visual Insights", layout="wide")

# =============================================================================
# Theme Globals (pulled from session set in Home page; fallback to light)
# =============================================================================
THEME = st.session_state.get("theme", "light")
ACCENT_COLOR = "#3a7bd5" if THEME == "light" else "#00d2ff"
BG_CARD = "#f9f9f9" if THEME == "light" else "rgba(255,255,255,0.05)"
TEXT_CARD = "#000" if THEME == "light" else "#fff"
PLOT_BG = "#ffffff" if THEME == "light" else "#1b1b1b"
PAPER_BG = PLOT_BG
GRIDCOLOR = "rgba(0,0,0,0.1)" if THEME == "light" else "rgba(255,255,255,0.1)"
FONT_COLOR = TEXT_CARD

# Custom Plotly template (dict is fine)
CUSTOM_TEMPLATE = {
    "layout": {
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor": PLOT_BG,
        "font": {"color": FONT_COLOR},
        "xaxis": {"gridcolor": GRIDCOLOR, "zerolinecolor": GRIDCOLOR},
        "yaxis": {"gridcolor": GRIDCOLOR, "zerolinecolor": GRIDCOLOR},
        "legend": {"bgcolor": PAPER_BG},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    }
}
px.defaults.template = CUSTOM_TEMPLATE

# =============================================================================
# CSS — reduce layout jump, style hero/cards
# =============================================================================
st.markdown(
    f"""
    <style>
    .vi-hero {{
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        margin-bottom: 1.5rem;
    }}
    .vi-hero h1 {{ margin: 0; font-size: 1.8rem; }}
    .vi-hero p {{ margin: 0.25rem 0 0 0; font-size: 1rem; opacity: 0.95; }}
    .suggestion-card {{
        border: 1px solid #ddd;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        background: {BG_CARD};
        color: {TEXT_CARD};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}
    div[data-testid="stPlotlyChart"] {{
        min-height: 400px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Hero Header
# =============================================================================
st.markdown(
    """
    <div class="vi-hero">
        <h1>📊 Visual Insights Dashboard</h1>
        <p>Interactive charts & smart (rule-based) suggestions for your dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Data Gate — Require upstream data from Home page
# =============================================================================
if "data" not in st.session_state:
    st.warning("⚠️ No data found. Please upload a CSV on the Home page first.")
    st.stop()

# Work on a *copy* of upstream data so we don't mutate original
_orig_df = st.session_state["data"]

# =============================================================================
# Data Helper Functions
# =============================================================================
def coerce_numeric_columns(df: pd.DataFrame, min_valid_ratio: float = 0.2) -> pd.DataFrame:
    """Coerce object-ish numeric columns (remove currency symbols, commas) to numbers."""
    dfc = df.copy()
    for col in dfc.columns:
        if dfc[col].dtype == "object":
            cleaned = dfc[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().mean() >= min_valid_ratio:
                dfc[col] = num
    return dfc


def detect_date_columns(df: pd.DataFrame, min_valid_ratio: float = 0.2) -> List[str]:
    """Heuristically detect date/time columns by name or parse success."""
    candidates: List[str] = []
    for col in df.columns:
        name_hit = any(k in col.lower() for k in ["date", "time", "dt"])
        converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        ratio = converted.notna().mean()
        if name_hit or ratio >= min_valid_ratio:
            if ratio > 0:
                candidates.append(col)
    return candidates


def fig_to_png_bytes(fig: go.Figure) -> BytesIO:
    """Export Plotly fig to PNG if kaleido available; fallback HTML."""
    buf = BytesIO()
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", engine="kaleido")
        buf.write(png_bytes)
        buf.seek(0)
        return buf
    except Exception:
        html_bytes = fig.to_html(full_html=False).encode("utf-8")
        buf.write(html_bytes)
        buf.seek(0)
        return buf

# =============================================================================
# Clean & Detect Types
# =============================================================================
df = coerce_numeric_columns(_orig_df)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
date_cols = detect_date_columns(df)

# "Safe" filtered lists (drop stale names — defensive)
_numeric_safe = [c for c in numeric_cols if c in df.columns]
_cat_safe     = [c for c in cat_cols     if c in df.columns]
_date_safe    = [c for c in date_cols    if c in df.columns]

# =============================================================================
# Auto Cache Reset When Data Changes
# =============================================================================
# Build a quick hash of current df content (cheap but sufficient for typical sizes)
# Using to_json for deterministic hash; large frames OK on HF typical sizes.
_current_hash = pd.util.hash_pandas_object(df, index=True).sum()

if "data_hash" not in st.session_state or st.session_state.data_hash != _current_hash:
    # Data changed — clear chart caches to avoid stale KeyError (e.g., 'calories')
    st.session_state.data_hash = _current_hash
    st.session_state.numeric_cache = {}
    st.session_state.categorical_cache = {}
    st.session_state.corr_cache = {}
    st.session_state.trend_cache = {}
    st.session_state.ai_cache = {}
    # do not clear _last_fig — user may still want last chart downloaded

# =============================================================================
# Optional tone indicator (not used in charts but useful display)
# =============================================================================
current_tone = st.session_state.get("tone", "formal").capitalize()
st.caption(f"Current storytelling tone: {current_tone}")

# =============================================================================
# Session State Initialization (safe re-init if not present)
# =============================================================================
if "numeric_cache" not in st.session_state:
    st.session_state.numeric_cache = {}
if "categorical_cache" not in st.session_state:
    st.session_state.categorical_cache = {}
if "corr_cache" not in st.session_state:
    st.session_state.corr_cache = {}
if "trend_cache" not in st.session_state:
    st.session_state.trend_cache = {}
if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}
if "_last_fig" not in st.session_state:
    st.session_state._last_fig = None
if "_last_fig_label" not in st.session_state:
    st.session_state._last_fig_label = None

# =============================================================================
# Cached Computations (data level — not figure level)
# =============================================================================
def get_trend_df(df_in: pd.DataFrame, date_col: str, num_cols: Tuple[str, ...], agg: str) -> pd.DataFrame:
    """
    Build a time-indexed numeric DataFrame suitable for trend plotting.
    Returns empty DataFrame if:
      - date col can't parse
      - no valid numeric cols remain
      - no rows after cleaning
    """
    if not num_cols:
        return pd.DataFrame()

    if date_col not in df_in.columns:
        return pd.DataFrame()

    # Parse date column
    df_ts = df_in.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
    df_ts = df_ts.dropna(subset=[date_col])
    if df_ts.empty:
        return pd.DataFrame()

    # Keep only valid numeric columns
    valid_cols = [c for c in num_cols if c in df_ts.columns]
    if not valid_cols:
        return pd.DataFrame()   # <-- FIX: return early instead of slicing

    # Force numeric dtype
    df_ts[valid_cols] = df_ts[valid_cols].apply(pd.to_numeric, errors="coerce")
    if df_ts[valid_cols].isna().all().all():
        return pd.DataFrame()

    # Build time index and resample
    dt = df_ts.set_index(date_col).sort_index()

    # If still no valid numeric columns after processing
    if dt.empty or all(col not in dt.columns for col in valid_cols):
        return pd.DataFrame()

    # Subset to valid numeric cols
    dt = dt[valid_cols]

    # Resample by aggregation level
    if agg == "Day":
        dt = dt.resample("D").mean()
    elif agg == "Month":
        dt = dt.resample("M").mean()
    elif agg == "Year":
        dt = dt.resample("Y").mean()

    return dt



# =============================================================================
# Plotly Chart Builders
# =============================================================================
def _apply_common_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title)
    return fig


def build_numeric_figs(col_name: str) -> Tuple[go.Figure, go.Figure, pd.Series]:
    """Return (hist_fig, box_fig, summary_series) for numeric column."""
    col_data = df[col_name].dropna()

    # Histogram
    fig_hist = px.histogram(
        col_data,
        x=col_name,
        nbins=30,
        opacity=0.85,
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_hist.update_traces(marker_line_width=0)
    fig_hist = _apply_common_layout(fig_hist, f"Distribution of {col_name}")

    # Boxplot
    fig_box = px.box(
        col_data,
        y=col_name,
        points="outliers",
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_box = _apply_common_layout(fig_box, f"{col_name} Boxplot")

    summary = col_data.describe()
    return fig_hist, fig_box, summary


def build_categorical_figs(col_name: str, top_n: int) -> Tuple[Optional[go.Figure], pd.DataFrame, Optional[go.Figure]]:
    """Return (bar_fig, table_df, pie_fig) for categorical column; figs may be None."""
    counts = df[col_name].value_counts().head(top_n)
    if counts.empty:
        return None, pd.DataFrame(columns=["Category", "Count", "Percent"]), None

    pct = counts / counts.sum() * 100
    table_df = pd.DataFrame({
        "Category": counts.index,
        "Count": counts.values,
        "Percent": pct.round(2).values
    })

    # Bar chart
    fig_bar = px.bar(
        table_df,
        x="Category",
        y="Count",
        text="Count",
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(xaxis_tickangle=-45)
    fig_bar = _apply_common_layout(fig_bar, f"Top {top_n} Categories: {col_name}")

    # Pie chart
    fig_pie = px.pie(
        table_df,
        names="Category",
        values="Count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.0,
    )
    fig_pie = _apply_common_layout(fig_pie, f"{col_name} Share")

    return fig_bar, table_df, fig_pie


@st.cache_data(show_spinner=False)
def compute_corr(df_sub: pd.DataFrame, method: str) -> pd.DataFrame:
    """Compute and cache correlation matrix for numeric columns."""
    if df_sub.empty or len(df_sub.columns) < 2:
        return pd.DataFrame()
    return df_sub.corr(method=method)


def build_corr_fig(method: str) -> Tuple[pd.DataFrame, go.Figure]:
    """Compute correlation matrix and return (DataFrame, Heatmap Figure)."""
    corr_df = compute_corr(df[_numeric_safe], method)

    if corr_df.empty:
        # Empty placeholder fig
        fig = go.Figure()
        fig.update_layout(title="No correlation data available")
        return corr_df, fig

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="corr"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig = _apply_common_layout(fig, f"Correlation ({method.title()})")
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    return corr_df, fig


def build_trend_fig(date_col: str, num_cols_sel: Tuple[str, ...], agg_choice: str) -> Optional[go.Figure]:
    """Return a Plotly line chart for one or more numeric columns over time; None if no data."""
    trend_df = get_trend_df(df, date_col, num_cols_sel, agg_choice)
    if trend_df.empty or trend_df.isna().all().all():
        return None

    trend_df = trend_df.reset_index().rename(columns={date_col: "_date_"})
    if len(num_cols_sel) > 1:
        trend_long = trend_df.melt(
            id_vars="_date_",
            value_vars=[c for c in num_cols_sel if c in trend_df.columns],
            var_name="Series",
            value_name="Value",
        )
        fig = px.line(
            trend_long,
            x="_date_",
            y="Value",
            color="Series",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
    else:
        # single series
        num_col = next((c for c in num_cols_sel if c in trend_df.columns), None)
        if num_col is None:
            return None
        fig = px.line(
            trend_df,
            x="_date_",
            y=num_col,
            color_discrete_sequence=[ACCENT_COLOR],
        )

    fig = _apply_common_layout(fig, f"Trend: {', '.join(num_cols_sel)} over {date_col} ({agg_choice})")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Value")
    return fig

# =============================================================================
# Tabs
# =============================================================================
tab_num, tab_cat, tab_corr, tab_time, tab_ai = st.tabs(
    ["📈 Numeric Explorer", "🔠 Categorical Explorer", "📉 Correlation", "⏱ Trends Over Time", "🤖 AI Chart Suggestions"]
)

# -----------------------------------------------------------------------------  
# Numeric Explorer Tab  
# -----------------------------------------------------------------------------  
with tab_num:
    st.subheader("📈 Numeric Explorer")

    if _numeric_safe:
        col_select = st.selectbox("Select a numeric column", _numeric_safe, key="num_sel_main")

        if col_select not in st.session_state.numeric_cache:
            with st.spinner("Generating charts..."):
                st.session_state.numeric_cache[col_select] = build_numeric_figs(col_select)

        fig_hist, fig_box, summary = st.session_state.numeric_cache[col_select]

        colH, colB = st.columns([2, 1])
        with colH:
            st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False}, key=f"num_hist_{col_select}")
        with colB:
            st.plotly_chart(fig_box, use_container_width=True, config={"displaylogo": False}, key=f"num_box_{col_select}")


        st.markdown("#### Summary Statistics")
        st.write(summary)

        st.session_state._last_fig = fig_hist
        st.session_state._last_fig_label = f"{col_select}_hist.png"
    else:
        st.info("No numeric columns detected.")

# -----------------------------------------------------------------------------  
# Categorical Explorer Tab  
# -----------------------------------------------------------------------------  
with tab_cat:
    st.subheader("🔠 Categorical Explorer")

    if _cat_safe:
        col_cat = st.selectbox("Select a categorical column", _cat_safe, key="cat_sel_main")
        top_n = st.slider("Top categories to display", 3, 20, 10, key="cat_topn")

        cache_key = (col_cat, top_n)
        if cache_key not in st.session_state.categorical_cache:
            with st.spinner("Summarizing categories..."):
                st.session_state.categorical_cache[cache_key] = build_categorical_figs(col_cat, top_n)

        fig_bar, table_df, fig_pie = st.session_state.categorical_cache[cache_key]

        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**Top {top_n} Categories**")
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False}, key=f"cat_bar_{col_cat}_{top_n}")
        else:
            st.warning("No data available for this chart.")
        with colB:
            st.markdown("**Counts & Percentages**")
            st.dataframe(table_df, use_container_width=True)

        show_pie = st.checkbox("Show Pie Chart", value=False, key="cat_show_pie")
        if show_pie and fig_pie is not None:
            st.plotly_chart(fig_pie, use_container_width=False, config={"displaylogo": False}, key=f"cat_pie_{col_cat}_{top_n}")

            st.session_state._last_fig = fig_pie
            st.session_state._last_fig_label = f"{col_cat}_pie.png"
        else:
            if fig_bar is not None:
                st.session_state._last_fig = fig_bar
                st.session_state._last_fig_label = f"{col_cat}_bar.png"
    else:
        st.info("No categorical columns detected.")

# -----------------------------------------------------------------------------  
# Correlation Tab  
# -----------------------------------------------------------------------------  
with tab_corr:
    st.subheader("📉 Correlation Heatmap")

    if len(_numeric_safe) >= 2:
        method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, key="corr_method")

        if method not in st.session_state.corr_cache:
            with st.spinner("Computing correlation..."):
                st.session_state.corr_cache[method] = build_corr_fig(method)

        corr_df, corr_fig = st.session_state.corr_cache[method]
        st.plotly_chart(corr_fig, use_container_width=True, config={"displaylogo": False}, key=f"corr_{method}")

        st.session_state._last_fig = corr_fig
        st.session_state._last_fig_label = f"corr_{method}.png"

        with st.expander("Show correlation matrix data"):
            st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")

# -----------------------------------------------------------------------------  
# Trends Over Time Tab  
# -----------------------------------------------------------------------------  
with tab_time:
    st.subheader("⏱ Trends Over Time")

    if _date_safe and _numeric_safe:
        date_col = st.selectbox("Select date/time column", _date_safe, key="time_date_sel")

        num_choices = st.multiselect(
            "Select numeric columns to trend",
            options=_numeric_safe,
            default=_numeric_safe[:1],
            key="time_num_select",
        )

        # Drop stale numeric selections (protect against KeyError)
        num_choices = [c for c in num_choices if c in df.columns]

        if not num_choices:
            st.warning("Selected numeric columns are no longer available; please reselect.")
        else:
            agg_choice = st.selectbox(
                "Aggregate by",
                ["Raw", "Day", "Month", "Year"],
                index=1,
                key="time_agg_sel"
            )

            # Cache key based on date + columns + agg
            trend_key = (date_col, tuple(num_choices), agg_choice)
            if trend_key not in st.session_state.trend_cache:
                with st.spinner("Building trend chart..."):
                    st.session_state.trend_cache[trend_key] = build_trend_fig(
                        date_col, tuple(num_choices), agg_choice
                    )

            trend_fig = st.session_state.trend_cache[trend_key]
            if trend_fig is None:
                st.warning("No data available for the selected aggregation.")
            else:
                st.plotly_chart(
                    trend_fig,
                    use_container_width=True,
                    config={"displaylogo": False},
                    key=f"trend_{date_col}_{'_'.join(valid_num_choices)}_{agg_choice}"
                )

                st.session_state._last_fig = trend_fig
                st.session_state._last_fig_label = f"trend_{date_col}_{agg_choice}.png"
    else:
        st.info("No suitable date and numeric columns detected.")

# -----------------------------------------------------------------------------  
# AI Chart Suggestions Tab (Rule-Based, Cached)  
# -----------------------------------------------------------------------------  
with tab_ai:
    st.subheader("🤖 AI Chart Recommendations (Rule-based)")
    st.write("These suggestions are based on your dataset and general visualization best practices.")

    suggestions = []
    if _numeric_safe:
        suggestions.append({
            "title": "Distribution of Numeric Data",
            "desc": f"Histograms help spot outliers in {', '.join(_numeric_safe[:3])}…",
            "action": "hist"
        })
    if len(_numeric_safe) > 1:
        suggestions.append({
            "title": "Correlation Between Numeric Columns",
            "desc": "Explore relationships using a heatmap.",
            "action": "corr"
        })
    if _cat_safe:
        suggestions.append({
            "title": "Most Frequent Categories",
            "desc": f"Visualize top categories from {_cat_safe[0]}.",
            "action": "cat_bar"
        })
    if _date_safe and _numeric_safe:
        suggestions.append({
            "title": "Trend Over Time",
            "desc": f"Line chart for {_numeric_safe[0]} over {_date_safe[0]}.",
            "action": "trend"
        })

    if not suggestions:
        st.info("No suggestions available for this dataset.")
    else:
        for s in suggestions:
            with st.expander(s["title"], expanded=False):
                st.markdown(f"<div class='suggestion-card'><p>{s['desc']}</p></div>", unsafe_allow_html=True)

                if s["action"] == "hist" and _numeric_safe:
                    ai_num = st.selectbox("Select numeric column", _numeric_safe, key=f"ai_hist_sel_{s['title']}")
                    if st.button("Generate Histogram", key=f"ai_hist_btn_{s['title']}"):
                        figH, figB, _ = build_numeric_figs(ai_num)
                        st.plotly_chart(figH, use_container_width=True, config={"displaylogo": False}, key=f"ai_hist_{ai_num}")
                        st.session_state.ai_cache["hist"] = (figH, f"{ai_num}_hist.png")
                        st.session_state._last_fig = figH
                        st.session_state._last_fig_label = f"{ai_num}_hist.png"

                elif s["action"] == "corr" and len(_numeric_safe) > 1:
                    ai_method = st.radio("Method", ["pearson", "spearman"], key=f"ai_corr_method_{s['title']}")
                    if st.button("Generate Correlation", key=f"ai_corr_btn_{s['title']}"):
                        _, figC = build_corr_fig(ai_method)
                        st.plotly_chart(figC, use_container_width=True, config={"displaylogo": False}, key=f"ai_corr_{ai_method}")
                        st.session_state.ai_cache["corr"] = (figC, f"corr_{ai_method}.png")
                        st.session_state._last_fig = figC
                        st.session_state._last_fig_label = f"corr_{ai_method}.png"

                elif s["action"] == "cat_bar" and _cat_safe:
                    ai_cat = st.selectbox("Select categorical column", _cat_safe, key=f"ai_cat_sel_{s['title']}")
                    ai_topn = st.slider("Top categories", 3, 20, 10, key=f"ai_cat_topn_{s['title']}")
                    if st.button("Generate Category Chart", key=f"ai_cat_btn_{s['title']}"):
                        fig_bar, table_df, fig_pie = build_categorical_figs(ai_cat, ai_topn)
                        if fig_bar is not None:
                            st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False}, key=f"ai_cat_bar_{ai_cat}_{ai_topn}")
                            st.dataframe(table_df, use_container_width=True)
                            st.session_state.ai_cache["cat_bar"] = (fig_bar, f"{ai_cat}_top{ai_topn}_bar.png")
                            st.session_state._last_fig = fig_bar
                            st.session_state._last_fig_label = f"{ai_cat}_top{ai_topn}_bar.png"
                        else:
                            st.warning("No data for selected categorical column.")

                elif s["action"] == "trend" and _date_safe and _numeric_safe:
                    ai_date = st.selectbox("Select date column", _date_safe, key=f"ai_trend_date_{s['title']}")
                    ai_num2 = st.selectbox("Select numeric column", _numeric_safe, key=f"ai_trend_num_{s['title']}")
                    ai_agg = st.selectbox("Aggregate by", ["Raw", "Day", "Month", "Year"], key=f"ai_trend_agg_{s['title']}")
                    if st.button("Generate Trend", key=f"ai_trend_btn_{s['title']}"):
                        figT = build_trend_fig(ai_date, (ai_num2,), ai_agg)
                        if figT is not None:
                            st.plotly_chart(figT, use_container_width=True, config={"displaylogo": False}, key=f"ai_trend_{ai_date}_{ai_num2}_{ai_agg}")
                            st.session_state.ai_cache["trend"] = (figT, f"{ai_num2}_over_{ai_date}_{ai_agg}.png")
                            st.session_state._last_fig = figT
                            st.session_state._last_fig_label = f"{ai_num2}_over_{ai_date}_{ai_agg}.png"
                        else:
                            st.warning("No data for selected trend.")

# =============================================================================
# Global Download Button
# =============================================================================
if st.session_state._last_fig is not None:
    buf = fig_to_png_bytes(st.session_state._last_fig)

    # crude detect HTML fallback
    is_html_fallback = False
    try:
        head = buf.getvalue()[:15]
        if head.strip().startswith(b"<"):
            is_html_fallback = True
    except Exception:
        pass

    st.download_button(
        label=f"⬇️ Download Last Chart ({'PNG' if not is_html_fallback else 'HTML'})",
        data=buf,
        file_name=st.session_state._last_fig_label or ("chart.html" if is_html_fallback else "chart.png"),
        mime="text/html" if is_html_fallback else "image/png",
    )
else:
    st.caption("No chart generated yet.")

