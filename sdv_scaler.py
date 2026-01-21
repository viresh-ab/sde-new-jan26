from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def scale_structured_data(df, rows):
    if df.empty:
        return df

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)

    return synthesizer.sample(rows)
