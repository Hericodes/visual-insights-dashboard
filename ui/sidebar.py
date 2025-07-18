"""
Sidebar UI for Data Storyteller.

Centralizes all sidebar layout, controls, and styling so the main app & pages
stay clean. Update *here* to change branding, theme, navigation, or dataset tools.
"""

from __future__ import annotations

import io
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st

from ui.theme import get_theme_colors


# ------------------------------------------------------------------
# Internal: CSS
# ------------------------------------------------------------------
def _render_sidebar_css() -> None:
    """Inject CSS styles for sidebar (theme-aware where possible)."""
    colors = get_theme_colors()
    text = colors["TEXT_CARD"]

    st.markdown(
        f"""
        <style>
        /* Sidebar gradient */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #001f3f 0%, #0074D9 100%);
            color: white;
        }}
        /* Force white text inside sidebar elements */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {{
            color: white !important;
        }}
        /* Sidebar buttons */
        section[data-testid="stSidebar"] .stButton>button {{
            background: white !important;
            color: #001f3f !important;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }}
        /* Mini badge pills */
        .ds-pill {{
            display:inline-block;
            padding:2px 8px;
            border-radius:8px;
            font-size:11px;
            font-weight:600;
            background:rgba(255,255,255,0.25);
            margin-left:4px;
        }}
        /* Small muted text */
        .ds-muted {{
            font-size:11px;
            opacity:0.75;
        }}
        /* Divider w/ fade */
        .ds-divider {{
            height:1px;
            width:100%;
            margin:0.5rem 0 0.75rem 0;
            background:rgba(255,255,255,0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return CSV bytes for download buttons."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _sidebar_branding(title: str = "📊 Data Storyteller", subtitle: str = "Turn data into stories") -> None:
    st.markdown(
        f"""
        <div style="text-align:center; padding-bottom: 8px;">
            <h2 style="margin-bottom:0;">{title}</h2>
            <p style="font-size:13px; margin-top:4px; opacity:0.8;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------
def render_sidebar(
    app_pages: Optional[List[Dict[str, str]]] = None,
    show_file_uploader: bool = False,
    sample_data: Optional[pd.DataFrame] = None,
    show_export: bool = True,
    show_env_badge: bool = True,
) -> Dict[str, Any]:
    """
    Render the sidebar UI and return structured info about user actions.

    Parameters
    ----------
    app_pages : list[dict], optional
        Navigation items. Each: {"path": str, "label": str, "icon": "🏠"}
    show_file_uploader : bool
        Include CSV uploader in the sidebar? (False recommended; main body is better UX.)
    sample_data : pd.DataFrame, optional
        If provided, "Load Sample Data" button appears.
    show_export : bool
        If True, show "Download Current Data" when dataset is loaded.
    show_env_badge : bool
        If True, show small environment note (e.g., “Live on Streamlit Cloud”).

    Returns
    -------
    dict
        {
          "theme": <"light"|"dark">,
          "tone": <"formal"|"friendly"|"analytical">,
          "uploaded_file": UploadedFile | None,
          "loaded_sample": bool,
          "cleared_data": bool,
          "export_clicked": bool,
        }
    """
    _render_sidebar_css()

    loaded_sample = False
    cleared_data = False
    export_clicked = False
    uploaded_file = None

    with st.sidebar:

        # Branding header
        _sidebar_branding()

        st.markdown('<div class="ds-divider"></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Navigation (collapsible)
        # ------------------------------------------------------------------
        with st.expander("🧭 Navigation", expanded=True):
            if app_pages:
                for pg in app_pages:
                    st.page_link(pg["path"], label=pg["label"], icon=pg.get("icon", ""))
            else:
                st.caption("No pages registered.")

        st.markdown('<div class="ds-divider"></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Data Tools (collapsible)
        # ------------------------------------------------------------------
        with st.expander("📦 Data Tools", expanded=True):

            # Sample loader
            if sample_data is not None:
                if st.button("📂 Load Sample Data", key="sb_load_sample"):
                    st.session_state["data"] = sample_data.copy()
                    st.session_state["data_source_name"] = "sample_data.csv"
                    loaded_sample = True
                    st.success("Sample data loaded.")

            # File uploader (optional inclusion)
            if show_file_uploader:
                uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="sb_uploader")

            # Clear data
            if st.button("🧹 Clear Data", key="sb_clear_data"):
                for k in ["data", "data_source_name"]:
                    st.session_state.pop(k, None)
                cleared_data = True
                st.experimental_rerun()

        st.markdown('<div class="ds-divider"></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Appearance / Tone (collapsible)
        # ------------------------------------------------------------------
        with st.expander("🎨 Appearance & Tone", expanded=False):
            # Theme
            theme_choice = st.radio(
                "Theme",
                ["🌞 Light", "🌙 Dark"],
                horizontal=True,
                index=0 if st.session_state.get("theme", "light") == "light" else 1,
                key="sb_theme_radio",
            )
            st.session_state["theme"] = "light" if theme_choice.startswith("🌞") else "dark"

            # Tone
            tone_map_ui_to_state = {"Formal": "formal", "Friendly": "friendly", "Analytical": "analytical"}
            tone_label = {v: k for k, v in tone_map_ui_to_state.items()}.get(
                st.session_state.get("tone", "formal"), "Formal"
            )
            tone_choice = st.selectbox(
                "🗣 Summary Tone",
                list(tone_map_ui_to_state.keys()),
                index=list(tone_map_ui_to_state.keys()).index(tone_label),
                key="sb_tone_select",
            )
            st.session_state["tone"] = tone_map_ui_to_state[tone_choice]

        st.markdown('<div class="ds-divider"></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Dataset Info + Export
        # ------------------------------------------------------------------
        with st.expander("📋 Dataset Info", expanded=True):
            if "data" in st.session_state:
                df_sidebar = st.session_state["data"]
                name = st.session_state.get("data_source_name", "Loaded Data")
                st.markdown(f"**Source:** {name}")
                st.success(f"Rows: **{len(df_sidebar)}**, Cols: **{len(df_sidebar.columns)}**")

                # Quick preview head toggle
                if st.checkbox("Preview first 5 rows", key="sb_preview_head"):
                    st.dataframe(df_sidebar.head(), use_container_width=True)

                if show_export:
                    csv_bytes = _df_to_csv_bytes(df_sidebar)
                    st.download_button(
                        label="⬇️ Download Current Data (CSV)",
                        data=csv_bytes,
                        file_name=name if name.endswith(".csv") else f"{name}.csv",
                        mime="text/csv",
                        key="sb_export_csv",
                    )
            else:
                st.info("No data loaded yet.")

        st.markdown('<div class="ds-divider"></div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # About & Links
        # ------------------------------------------------------------------
        with st.expander("ℹ️ About & Links", expanded=False):
            st.write(
                "Data Storyteller helps you analyze, visualize, and narrate datasets in minutes. "
                "Built with Streamlit."
            )
            if show_env_badge:
                st.markdown('<span class="ds-pill">Live</span>', unsafe_allow_html=True)
            st.markdown(
                """
                **Resources:**
                - Documentation (coming soon)
                - GitHub Repo (coming soon)
                - Contact: hello@example.com
                """.strip()
            )

    # Done
    return {
        "theme": st.session_state["theme"],
        "tone": st.session_state["tone"],
        "uploaded_file": uploaded_file,
        "loaded_sample": loaded_sample,
        "cleared_data": cleared_data,
        "export_clicked": export_clicked,
    }
