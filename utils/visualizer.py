# utils/visualizer.py

import streamlit as st
import pandas as pd

def show_charts(df: pd.DataFrame, analysis: dict):
    numeric = analysis.get("numeric_summary", {})
    categories = analysis.get("top_categories", {})
    date_cols = analysis.get("date_columns", [])

    st.subheader("📊 Visual Insights")

    # Bar charts for top categories
    if categories:
        st.markdown("**Top Category Visuals**")
        for col, top_vals in categories.items():
            st.bar_chart(pd.Series(top_vals), height=250)

    # Line charts for date vs numeric columns
    if date_cols and numeric:
        st.markdown("**Trends Over Time**")
        date_col = date_cols[0]  # use the first detected date column
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df_sorted = df.sort_values(date_col)

            for col in numeric.keys():
                st.line_chart(
                    df_sorted[[date_col, col]].dropna().set_index(date_col), height=300
                )
        except Exception as e:
            st.warning(f"Could not plot time series from `{date_col}`: {e}")
    else:
        st.info("No date + numeric columns available for time-based plots.")
