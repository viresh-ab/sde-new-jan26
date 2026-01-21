def validate_schema(real_df, synthetic_df):
    synthetic_df.columns = [c.strip() for c in synthetic_df.columns]

    for col in real_df.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = ""

    return synthetic_df[real_df.columns]
