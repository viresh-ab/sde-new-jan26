import pandas as pd
import os
import re
from io import StringIO
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BATCH_SIZE = 200


def generate_qa_synthetic_data(sample_df: pd.DataFrame, rows: int) -> pd.DataFrame:
    batches = []
    remaining = rows

    while remaining > 0:
        size = min(BATCH_SIZE, remaining)
        batches.append(_generate_batch(sample_df, size))
        remaining -= size

    return pd.concat(batches, ignore_index=True)


def _generate_batch(sample_df, rows):
    columns = list(sample_df.columns)
    examples = sample_df.sample(min(5, len(sample_df)), random_state=None)

    prompt = f"""
Generate synthetic survey answers.

STRICT RULES:
- Output CSV ONLY
- Every value must be wrapped in double quotes
- If a value contains a double quote, escape it as two double quotes ("")
- NEVER use unescaped "
- Header must be EXACTLY:
{columns}
- Generate EXACTLY {rows} rows
- Natural language, diverse answers
- Do NOT copy examples

Style reference:
{examples.to_csv(index=False)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )

    raw = response.choices[0].message.content

    # Remove markdown if present
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

    # 🔒 SAFETY FIX: normalize broken quotes
    raw = raw.replace('"""""', '""')  # over-escaped
    raw = raw.replace('"""', '""')

    try:
        df = pd.read_csv(
            StringIO(raw),
            quotechar='"',
            escapechar='"',
            engine="python"
        )
    except Exception as e:
        raise ValueError(
            f"CSV parsing failed.\n\nRAW OUTPUT:\n{raw}"
        ) from e

    return df[columns]
