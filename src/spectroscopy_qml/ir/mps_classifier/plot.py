import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/raw")
ERROR_FILE = Path("src/spectroscopy_qml/ir/mps_classifier/results_per_label/error_analysis/sample_errors.csv")  # or CNN / TTN

errors = pd.read_csv(ERROR_FILE)

# choose examples manually or by highest error_count
examples = errors.sort_values("error_count", ascending=False).head(3)

wanted_ids = set(examples["sample_index"].astype(int))

row_offset = 0
spectra = {}

for parquet_file in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(parquet_file, columns=["ir_spectra", "smiles"])
    df["_row_id"] = np.arange(row_offset, row_offset + len(df), dtype=np.int64)
    row_offset += len(df)

    hits = df[df["_row_id"].isin(wanted_ids)]
    for _, row in hits.iterrows():
        spectra[int(row["_row_id"])] = np.asarray(row["ir_spectra"], dtype=float)

for _, row in examples.iterrows():
    sample_id = int(row["sample_index"])
    spectrum = spectra[sample_id]

    plt.figure(figsize=(8, 3))
    plt.plot(spectrum)
    plt.title(f"Sample {sample_id}")
    plt.xlabel("Spectral position")
    plt.ylabel("Intensity")

    text = (
        f"True: {row['true_labels']}\n"
        f"Pred: {row['predicted_labels']}\n"
        f"FN: {row['false_negative_labels']}\n"
        f"FP: {row['false_positive_labels']}"
    )

    plt.gcf().text(0.02, -0.25, text, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"spectrum_error_{sample_id}.png", dpi=300, bbox_inches="tight")
    plt.close()