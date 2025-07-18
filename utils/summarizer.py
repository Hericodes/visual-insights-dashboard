# utils/summarizer.py

def generate_summary(analysis: dict, tone: str = "formal") -> str:
    shape = analysis.get("shape", {})
    rows = shape.get("rows", 0)
    cols = shape.get("columns", 0)

    numeric = analysis.get("numeric_summary", {})
    categories = analysis.get("top_categories", {})
    date_cols = analysis.get("date_columns", [])
    col_types = analysis.get("column_types", {})

    # Tone definitions
    if tone == "friendly":
        intro = f"Hey there! 👋 You've got a dataset with {rows} rows and {cols} columns — nice!"
        numeric_text = "Let's peek into some numbers:"
        category_text = "Here are the top repeated values:"
        date_text = "We also found some dates you might want to explore:"
        type_text = "Here's what kind of data you've got:"
    elif tone == "analytical":
        intro = f"This dataset consists of {rows} records across {cols} dimensions. Here's a statistical overview:"
        numeric_text = "Numerical breakdown:"
        category_text = "Dominant categorical distributions:"
        date_text = "Time-based fields detected:"
        type_text = "Data types identified:"
    else:  # formal
        intro = f"The dataset contains {rows} rows and {cols} columns."
        numeric_text = "Summary of numerical features:"
        category_text = "Most frequent values in categorical columns:"
        date_text = "Detected date/time columns:"
        type_text = "Detected column data types:"

    summary = f"### 📊 Summary\n\n{intro}\n\n"

    # Column types
    if col_types:
        summary += f"**{type_text}**\n"
        for col, dtype in col_types.items():
            summary += f"- `{col}`: {dtype}\n"
        summary += "\n"

    # Numeric columns
    if numeric:
        summary += f"**{numeric_text}**\n"
        for col, stats in numeric.items():
            summary += f"- `{col}`: Mean = {stats['mean']}, Min = {stats['min']}, Max = {stats['max']}, Std = {stats['std']}\n"
        summary += "\n"

    # Categorical columns
    if categories:
        summary += f"**{category_text}**\n"
        for col, top_vals in categories.items():
            items = ", ".join([f"{val} ({count})" for val, count in top_vals.items()])
            summary += f"- `{col}`: {items}\n"
        summary += "\n"

    # Date columns
    if date_cols:
        summary += f"**{date_text}**\n"
        for col in date_cols:
            summary += f"- `{col}`\n"

    return summary
