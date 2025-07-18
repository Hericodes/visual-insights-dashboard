# utils/analyzer.py

import pandas as pd
import re

def try_convert_to_numeric(series):
    # Remove non-numeric characters (commas, currency, etc.)
    cleaned = series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

def analyze_data(df: pd.DataFrame) -> dict:
    analysis = {}

    # Shape
    analysis['shape'] = {
        'rows': df.shape[0],
        'columns': df.shape[1]
    }

    # Column types (initial)
    analysis['column_types'] = df.dtypes.astype(str).to_dict()

    # Try to extract numeric data
    numeric_summary = {}
    for col in df.columns:
        converted = try_convert_to_numeric(df[col])
        if converted.notna().sum() > 0:
            numeric_summary[col] = {
                'mean': round(converted.mean(), 2),
                'min': round(converted.min(), 2),
                'max': round(converted.max(), 2),
                'std': round(converted.std(), 2)
            }

    analysis['numeric_summary'] = numeric_summary

    # Top categories (text columns)
    top_categories = {}
    for col in df.select_dtypes(include='object').columns:
        top = df[col].value_counts().head(3)
        top_categories[col] = top.to_dict()
    analysis['top_categories'] = top_categories

    # Detect date columns
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            continue
    analysis['date_columns'] = date_cols

    return analysis
