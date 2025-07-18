"""
Theme utilities for Data Storyteller.
Provides centralized theme color settings and dynamic gradients for light/dark mode.
"""

from __future__ import annotations
import streamlit as st

# -------------------------------------------------------------------
# Default color sets for light and dark themes
# -------------------------------------------------------------------

LIGHT_THEME = {
    "BACKGROUND": "#ffffff",
    "TEXT_CARD": "#000000",
    "BG_CARD": "#f9f9f9",
    "ACCENT_COLOR": "#3a7bd5",
    "GRADIENT": "linear-gradient(180deg, #F0F4F8 0%, #E8EEF5 100%)",
    "SIDEBAR_GRADIENT": "linear-gradient(180deg, #001f3f 0%, #0074D9 100%)",
}

DARK_THEME = {
    "BACKGROUND": "#121212",
    "TEXT_CARD": "#ffffff",
    "BG_CARD": "#1E1E1E",
    "ACCENT_COLOR": "#00d2ff",
    "GRADIENT": "linear-gradient(180deg, #1F1F1F 0%, #181818 100%)",
    "SIDEBAR_GRADIENT": "linear-gradient(180deg, #111 0%, #333 100%)",
}

# -------------------------------------------------------------------
# Main function to fetch colors dynamically
# -------------------------------------------------------------------
def get_theme_colors() -> dict:
    """
    Return a dict of theme colors based on current session_state['theme'].
    If no theme is set, defaults to light theme.
    """
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        return DARK_THEME
    return LIGHT_THEME


# -------------------------------------------------------------------
# Optional helper: inject global CSS (call once in app.py if needed)
# -------------------------------------------------------------------
def inject_global_theme_css() -> None:
    """
    Injects CSS variables for light/dark mode styling.
    Useful for global color consistency in all pages.
    """
    colors = get_theme_colors()
    st.markdown(
        f"""
        <style>
        :root {{
            --bg-main: {colors['BACKGROUND']};
            --text-card: {colors['TEXT_CARD']};
            --bg-card: {colors['BG_CARD']};
            --accent: {colors['ACCENT_COLOR']};
        }}
        html, body, [class*="css"] {{
            background-color: var(--bg-main) !important;
            color: var(--text-card) !important;
        }}
        /* Cards and Metrics */
        div[data-testid="stMetric"] {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
