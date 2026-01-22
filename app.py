import streamlit as st
import pandas as pd

from schema_extractor import extract_schema
from qa_llm_generator import generate_qa_synthetic_data
from sdv_scaler import scale_structured_data
from hybrid_merger import merge_hybrid
from validator import validate_schema


# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Markelytics AI | Synthetic Hybrid Studio",
    layout="wide"
)

# ----------------------------------------------------
# Hide Streamlit UI
# ----------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stViewerBadge"],
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Markelytics AI | Synthetic Hybrid Studio")
st.caption("ChatGPT (text) + SDV (structured) synthetic data generation")

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def split_columns(df):
    text_cols, structured_cols = [], []
    for col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > 40:
            text_cols.append(col)
        else:
            structured_cols.append(col)
    return text_cols, structured_cols


def ensure_df(obj, name):
    if obj is None:
        raise ValueError(f"{name} returned None")

    if isinstance(obj, list):
        obj = pd.DataFrame(obj)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{name} must return a DataFrame, got {type(obj)}")

    if obj.empty:
        raise ValueError(f"{name} returned an empty DataFrame")

    return obj


# ----------------------------------------------------
# Upload CSV
# ----------------------------------------------------
uploaded_file = st.file_uploader("Upload Survey CSV", type=["csv"])

if uploaded_file:
    real_df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")
    st.dataframe(real_df.head())

    final_rows = st.number_input(
        "Number of synthetic rows",
        min_value=50,
        max_value=10000,
        value=1000,
        step=50
    )

    if st.button("🚀 Generate Hybrid Synthetic Data"):
        with st.spinner("Generating hybrid synthetic data..."):
            try:
                schema = extract_schema(real_df)

                text_cols, structured_cols = split_columns(real_df)

                text_real = real_df[text_cols]
                structured_real = real_df[structured_cols]

                # ----------------------------
                # Generate synthetic components
                # ----------------------------
                text_syn = None
                structured_syn = None

                if not text_real.empty:
                    text_syn = ensure_df(
                        generate_qa_synthetic_data(text_real, final_rows),
                        "LLM generator"
                    )

                if not structured_real.empty:
                    structured_syn = ensure_df(
                        scale_structured_data(structured_real, final_rows),
                        "SDV generator"
                    )

                # ----------------------------
                # Merge + validate
                # ----------------------------
                final_df = ensure_df(
                    merge_hybrid(structured_syn, text_syn, final_rows),
                    "Hybrid merger"
                )

                final_df = ensure_df(
                    validate_schema(real_df, final_df),
                    "Schema validator"
                )

                st.success(f"Synthetic dataset generated ({len(final_df)} rows)")
                st.dataframe(final_df.head())

                st.download_button(
                    "⬇ Download Synthetic CSV",
                    final_df.to_csv(index=False),
                    "synthetic_hybrid_data.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error("Failed to generate synthetic data")
                st.code(str(e))
