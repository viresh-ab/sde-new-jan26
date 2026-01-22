import pandas as pd


def merge_hybrid(structured_df, text_df):
    """
    Safe hybrid merge:
    - If one side is empty or None, return the other
    - Never return empty unless both are empty
    """

    if structured_df is None or structured_df.empty:
        if text_df is None or text_df.empty:
            raise ValueError("Both structured and text data are empty")
        return text_df.reset_index(drop=True)

    if text_df is None or text_df.empty:
        return structured_df.reset_index(drop=True)

    min_len = min(len(structured_df), len(text_df))

    structured_df = structured_df.iloc[:min_len].reset_index(drop=True)
    text_df = text_df.iloc[:min_len].reset_index(drop=True)

    merged = pd.concat([structured_df, text_df], axis=1)

    if merged.empty:
        raise ValueError("Hybrid merge resulted in empty DataFrame")

    return merged
