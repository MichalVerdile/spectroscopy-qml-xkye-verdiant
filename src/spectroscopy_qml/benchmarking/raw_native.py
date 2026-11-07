"""Strict raw_native cohort: no imports from legacy signal preprocessing.

Parquet is streamed twice into float64 memory maps. Every model uses `features`
for the SAME checked float32 conversion and includes the explicit occupancy mask.
No value sorting, clipping, replacement, resampling or peak selection is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .raw_labels import LABEL_NAMES, molecule_metadata

RAW_SCHEMA = "raw_native_v1"
MODALITIES = ("cnmr", "hnmr", "ir", "msms_pos", "msms_neg")
COLUMNS = {
    "cnmr": ("c_nmr_spectra",),
    "hnmr": ("h_nmr_spectra",),
    "ir": ("ir_spectra",),
    "msms_pos": ("msms_positive_40ev", "msms_cfmid_positive_40ev"),
    "msms_neg": ("msms_negative_40ev", "msms_cfmid_negative_40ev"),
}
NATIVE_LENGTHS = {"cnmr": 10000, "hnmr": 10000, "ir": 1800}
CAST_RTOL = 1e-6


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    for start in range(0, len(array), 128):
        digest.update(np.ascontiguousarray(array[start : start + 128]).tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def checked_float32(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        result = values.astype(np.float32)
    # Relative-only tolerance also rejects underflow of a real nonzero peak to zero.
    if not np.isfinite(result).all() or not np.allclose(result, values, rtol=CAST_RTOL, atol=0):
        raise ValueError("Measurement cannot be represented in float32 within rtol=1e-6, atol=0")
    return result


def validate_signal(value, modality: str, expected_lengths: dict) -> np.ndarray:
    if value is None:
        raise ValueError(f"Missing {modality} signal")
    # Arrow nested lists sometimes arrive as an object array of pairs.
    try:
        array = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Malformed {modality} signal") from error
    if modality in NATIVE_LENGTHS:
        if array.ndim != 1 or len(array) != expected_lengths[modality]:
            raise ValueError(
                f"{modality}: observed {array.shape}, expected ({expected_lengths[modality]},)"
            )
        array = array[:, None]
    else:
        # Empty (observed) lists are valid; null signals are missing, not empty.
        if array.size == 0 and array.ndim == 1:
            array = array.reshape(0, 2)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{modality}: expected all original (m/z, intensity) pairs")
    if not np.isfinite(array).all():
        raise ValueError(f"{modality}: NaN/Inf or null measurement")
    checked_float32(array)
    return array


def parquet_rows(paths: list[Path], max_samples: int | None = None):
    import pyarrow.parquet as pq

    total = 0
    for path in paths:
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                raise ValueError(f"{path}: not materialized Parquet (possibly a Git LFS pointer)")
        parquet = pq.ParquetFile(path)
        columns = {}
        for modality, aliases in COLUMNS.items():
            matches = [name for name in aliases if name in parquet.schema_arrow.names]
            if len(matches) != 1:
                raise ValueError(f"{path}: require exactly one {modality} column, found {matches}")
            columns[modality] = matches[0]
        row_index = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["smiles", *columns.values()]):
            for row in batch.to_pylist():
                if max_samples is not None and total >= max_samples:
                    return
                yield path.name, row_index, row["smiles"], {m: row[c] for m, c in columns.items()}
                row_index += 1
                total += 1


@dataclass
class RawCohort:
    root: Path
    manifest: dict
    records: list[dict]
    labels: np.ndarray

    @classmethod
    def open(cls, root: Path, verify: bool = True) -> RawCohort:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] != RAW_SCHEMA:
            raise ValueError("Unsupported raw cohort schema")
        identity = {k: v for k, v in manifest.items() if k != "dataset_hash"}
        if digest_json(identity) != manifest["dataset_hash"]:
            raise ValueError("Raw manifest hash mismatch")
        if verify:
            for name, expected in manifest["files"].items():
                if file_sha256(root / name) != expected:
                    raise ValueError(f"Raw cache content changed: {name}")
        return cls(
            root,
            manifest,
            json.loads((root / "records.json").read_text()),
            np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False),
        )

    def values(self, modality: str) -> np.ndarray:
        return np.load(self.root / f"{modality}_values.npy", mmap_mode="r", allow_pickle=False)

    def masks(self, modality: str) -> np.ndarray:
        return np.load(self.root / f"{modality}_mask.npy", mmap_mode="r", allow_pickle=False)

    def features(self, modality: str, indices) -> np.ndarray:
        """Lossless layout: [raw channel(s), occupied]. Tree models flatten this.

        Dense: [intensity, 1]. MS/MS: [m/z, intensity, occupied]. Padding is
        [0, 0, 0], distinct even from a genuine measured (0, 0) peak [0, 0, 1].
        """
        raw = checked_float32(self.values(modality)[indices])
        mask = self.masks(modality)[indices].astype(np.float32)
        return np.concatenate([raw, mask[..., None]], axis=-1)

    def input_hash(self, modality: str) -> str:
        return self.manifest["modalities"][modality]["model_input_hash"]


def prepare_cohort(
    data_dir: Path,
    cache_dir: Path,
    *,
    metadata_path: Path | None = None,
    invalid_policy: str = "error",
    max_samples: int | None = None,
    expected_lengths: dict | None = None,
    synthetic: bool = False,
) -> RawCohort:
    """Validate all five modalities together; exclude invalid rows ONLY cohort-wide.

    `expected_lengths` overrides are reserved for explicitly synthetic fixtures.
    Padding length uses label-blind source shape metadata, never validation scores.
    """
    if invalid_policy not in {"error", "exclude"}:
        raise ValueError("invalid_policy must be error or exclude")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if expected_lengths is not None and not synthetic:
        raise ValueError("Native dimensions cannot be overridden for real data")
    lengths = expected_lengths or NATIVE_LENGTHS
    paths = sorted(Path(data_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")
    sources = [{"name": p.name, "sha256": file_sha256(p)} for p in paths]
    implementation = {
        p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name("raw_labels.py"))
    }
    spec = {
        "schema": RAW_SCHEMA,
        "sources": sources,
        "implementation": implementation,
        "invalid_policy": invalid_policy,
        "max_samples": max_samples,
        "synthetic": synthetic,
        "expected_lengths": lengths,
        "metadata_sha256": file_sha256(metadata_path) if metadata_path else None,
    }
    destination = Path(cache_dir) / digest_json(spec)
    if (destination / "manifest.json").exists():
        return RawCohort.open(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="raw-building-", dir=destination.parent))
    try:
        records, labels, rejected = [], [], []
        maxima = {m: lengths.get(m, 1) for m in MODALITIES}
        # First pass: metadata and validation only. No full spectra kept in RAM.
        for filename, row, smiles, signals in parquet_rows(paths, max_samples):
            try:
                canonical, group, scaffold, target = molecule_metadata(smiles)
                validated = {m: validate_signal(v, m, lengths) for m, v in signals.items()}
            except ValueError as error:
                rejection = {"file": filename, "row": row, "reason": str(error)}
                if invalid_policy == "error":
                    raise ValueError(f"Invalid sample {rejection}") from error
                rejected.append(rejection)
                continue
            dimensions = {m: len(v) for m, v in validated.items()}
            for m, size in dimensions.items():
                maxima[m] = max(maxima[m], size)
            records.append(
                {
                    "sample_id": digest_json([filename, row, canonical]),
                    "file": filename,
                    "row": row,
                    "canonical_smiles": canonical,
                    "molecule_id": digest_json(group),
                    "scaffold_id": digest_json(scaffold),
                    "original_dimensions": dimensions,
                }
            )
            labels.append(target)
        if not records:
            raise ValueError("No valid samples in the common five-modality cohort")
        arrays, masks = {}, {}
        for m in MODALITIES:
            channels = 1 if m in NATIVE_LENGTHS else 2
            arrays[m] = np.lib.format.open_memmap(
                staging / f"{m}_values.npy",
                mode="w+",
                dtype="<f8",
                shape=(len(records), maxima[m], channels),
            )
            masks[m] = np.lib.format.open_memmap(
                staging / f"{m}_mask.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(len(records), maxima[m]),
            )
            arrays[m][:] = 0
            masks[m][:] = False
        accepted = {(r["file"], r["row"]): i for i, r in enumerate(records)}
        for filename, row, _, signals in parquet_rows(paths, max_samples):
            index = accepted.get((filename, row))
            if index is None:
                continue
            for m, value in signals.items():
                signal = validate_signal(value, m, lengths)
                arrays[m][index, : len(signal)] = signal
                masks[m][index, : len(signal)] = True
        if sources != [{"name": p.name, "sha256": file_sha256(p)} for p in paths]:
            raise ValueError("Source changed during raw data preparation")
        modality_meta = {}
        metadata = json.loads(metadata_path.read_text()) if metadata_path else {}
        for m in MODALITIES:
            arrays[m].flush()
            masks[m].flush()
            h = hashlib.sha256()
            h.update(str((len(records), maxima[m], arrays[m].shape[2] + 1)).encode())
            for start in range(0, len(records), 64):
                packed = np.concatenate(
                    [
                        checked_float32(arrays[m][start : start + 64]),
                        masks[m][start : start + 64, :, None].astype(np.float32),
                    ],
                    axis=-1,
                )
                h.update(packed.tobytes())
            axis = metadata.get(COLUMNS[m][0], {}).get("dimensions")
            if axis is not None:
                axis = np.asarray(axis, dtype="<f8")
                if axis.shape != (maxima[m],) or not np.isfinite(axis).all():
                    raise ValueError(f"Physical axis does not match {m}")
                np.save(staging / f"{m}_axis.npy", axis)
            modality_meta[m] = {
                "shape": list(arrays[m].shape),
                "original_values_hash": array_sha256(arrays[m]),
                "mask_hash": array_sha256(masks[m]),
                "model_input_hash": h.hexdigest(),
                "axis": f"{m}_axis.npy"
                if axis is not None
                else ("first channel: m/z" if m.startswith("msms") else None),
                "length_min": min(r["original_dimensions"][m] for r in records),
                "length_max": max(r["original_dimensions"][m] for r in records),
            }
        # Close mmap handles before directory rename (Windows).
        arrays.clear()
        masks.clear()
        np.save(staging / "labels.npy", np.stack(labels))
        atomic_json(staging / "records.json", records)
        atomic_json(staging / "rejected.json", rejected)
        manifest = {
            **spec,
            "mode": "raw_native",
            "label_names": list(LABEL_NAMES),
            "sample_count": len(records),
            "rejected_count": len(rejected),
            "modalities": modality_meta,
            "precision": {"storage": "float64", "model": "float32", "rtol": CAST_RTOL, "atol": 0},
            "files": {p.name: file_sha256(p) for p in sorted(staging.iterdir())},
        }
        manifest["dataset_hash"] = digest_json(manifest)
        atomic_json(staging / "manifest.json", manifest)
        try:
            staging.rename(destination)
        except OSError:
            if not (destination / "manifest.json").exists():
                raise
            # A concurrent builder can only be reused after complete verification.
        return RawCohort.open(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
