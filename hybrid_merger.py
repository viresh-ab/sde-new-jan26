import pandas as pd


def validate_schema(real_df, synthetic_df):
    """
    Schema-safe validation:
    - Aligns columns
    - Preserves data even if some rows are invalid
    - Never returns empty unless input is empty
    """

    if synthetic_df is None or synthetic_df.empty:
        raise ValueError("Synthetic data is empty before validation")

    validated = synthetic_df.copy()

    # Keep only known columns
    validated = validated[[c for c in validated.columns if c in real_df.columns]]

    # Reorder columns to match real data
    validated = validated.reindex(columns=real_df.columns, fill_value=None)

    # Attempt type coercion (non-destructive)
    for col in real_df.columns:
        try:
            validated[col] = validated[col].astype(real_df[col].dtype)
        except Exception:
            pass  # keep best-effort values

    if validated.empty:
        raise ValueError("Validation removed all rows")

    return validated
