import pandas as pd


def merge_hybrid(structured_df, text_df):
    min_len = min(len(structured_df), len(text_df))
    structured_df = structured_df.iloc[:min_len].reset_index(drop=True)
    text_df = text_df.iloc[:min_len].reset_index(drop=True)
    return pd.concat([structured_df, text_df], axis=1)
