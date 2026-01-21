def extract_schema(df):
    return {col: str(df[col].dtype) for col in df.columns}
