#!/usr/bin/env python3
"""Validate and prepare a RunPod without altering the scientific source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from protocol import (MODALITIES, REFERENCE, SEEDS, check_science,
                      effective_memory_bytes, publish, sha256, stamp, write_json)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def import_trainer(repo):
    sys.path.insert(0, str(repo / "src"))
    spec = importlib.util.spec_from_file_location("xgb_reference", repo / "benchmark/xgb/scripts/run_xgb_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(root, local):
    import numpy as np
    import torch
    import xgboost as xgb
    import importlib.metadata as md

    mount = subprocess.check_output(["findmnt", "-J", "-T", str(root)], text=True)
    info = json.loads(mount)["filesystems"][0]
    if Path(info["target"]).resolve() != root.resolve():
        raise ValueError(f"{root} is not an independently mounted persistent filesystem: {info}")
    if info["fstype"] in ("overlay", "tmpfs"):
        raise ValueError("Refusing ephemeral output storage")
    probe = root / "experiments/xgboost_raw_native/configurations" / f"storage_probe_{stamp()}.json"
    write_json(probe, {"probe": "persistent write/close/reopen", "created": stamp()})
    chmod_supported = True
    try:
        probe.chmod(0o644)
    except OSError:
        chmod_supported = False
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python 3.12 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    if effective_memory_bytes() < 200 * 1024**3:
        raise RuntimeError("Insufficient effective RAM for the unchanged full-width loader")
    rng = np.random.default_rng(9041)
    X = rng.normal(size=(512, 10000)).astype(np.float32)
    y = np.arange(512) % 2
    smoke = xgb.XGBClassifier(n_estimators=3, tree_method="hist", device="cuda", max_depth=2, n_jobs=4)
    smoke.fit(X[:400], y[:400], eval_set=[(X[400:], y[400:])], verbose=False)
    config = json.loads(smoke.get_booster().save_config())
    device = config["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost smoke test used {device}")
    local_model = local / "gpu_smoke.ubj"
    smoke.save_model(local_model)
    publish(local_model, probe.parent / "gpu_smoke.ubj")
    record = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "xgboost_device": device, "effective_memory_bytes": effective_memory_bytes(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)), "mount": info,
        "chmod_supported": chmod_supported, "gpu_smoke": "passed, synthetic only",
        "packages": {name: md.version(name) for name in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "rdkit", "xgboost", "iterative-stratification"]},
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
    }
    write_json(probe.parent / "environment.json", record)
    print(json.dumps(record, indent=2), flush=True)


def dataset(repo, root):
    data = root / "data/raw"
    data.mkdir(parents=True, exist_ok=True)
    names = git(repo, "ls-tree", "-r", "--name-only", REFERENCE, "--", "data/raw").splitlines()
    names = sorted(name for name in names if name.endswith(".parquet"))
    if len(names) != 245:
        raise ValueError(f"Expected 245 Parquet files, got {len(names)}")
    entries = []
    missing = []
    for name in names:
        pointer = git(repo, "show", f"{REFERENCE}:{name}")
        fields = dict(line.split(" ", 1) for line in pointer.splitlines())
        if fields.get("version") != "https://git-lfs.github.com/spec/v1":
            raise ValueError(f"Not an expected LFS pointer: {name}")
        oid = fields["oid"].removeprefix("sha256:")
        entry = {"name": Path(name).name, "sha256": oid, "bytes": int(fields["size"])}
        entries.append(entry)
        target = data / entry["name"]
        if not target.exists() or target.stat().st_size != entry["bytes"] or sha256(target) != oid:
            missing.append(entry)
    if missing:
        print(f"Fetching reference LFS data; {len(missing)} persistent files missing", flush=True)
        subprocess.run(["git", "-C", str(repo), "lfs", "fetch", "origin", REFERENCE,
                        "--include=data/raw/*.parquet", "--exclude="], check=True)
        for index, entry in enumerate(missing, 1):
            oid = entry["sha256"]
            source = repo / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
            if source.stat().st_size != entry["bytes"] or sha256(source) != oid:
                raise ValueError(f"Corrupt LFS object: {entry['name']}")
            print(f"Publishing raw data {index}/{len(missing)} {entry['name']}", flush=True)
            publish(source, data / entry["name"])
    actual = {p.name for p in data.glob("*.parquet")}
    if actual != {e["name"] for e in entries}:
        raise ValueError("Unexpected Parquet files would change dataset ordering")
    manifest = root / "data/dataset_manifest.json"
    write_json(manifest, {"reference_commit": REFERENCE, "ordering": "lexicographic filename, original row order", "files": entries})
    return data, manifest


def audit_and_splits(repo, root, local, data, accept_reconstructed):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from rdkit import Chem

    trainer = import_trainer(repo)
    from spectroscopy_qml.benchmarking.splits import create_split_artifact, label_digest, load_split_artifact
    from spectroscopy_qml.benchmarking.preprocessing import parse_preprocessing, preprocess_spectra, make_msms_spectrum

    labels, identities = {m: [] for m in MODALITIES}, {m: [] for m in MODALITIES}
    rows = []
    sample_digests = {m: hashlib.sha256() for m in MODALITIES}
    audit = {m: {"native_lengths": {}, "nonfinite_values": 0, "excluded_rows": 0} for m in MODALITIES}
    columns = ["smiles"] + [entry[1] for entry in MODALITIES.values()]
    files = sorted(data.glob("*.parquet"))
    for number, path in enumerate(files, 1):
        print(f"AUDIT {number}/{len(files)} {path.name}", flush=True)
        with path.open("rb") as f:
            if f.read(4) != b"PAR1":
                raise ValueError(f"Not real Parquet data: {path}")
        metadata = pq.ParquetFile(path)
        missing = set(columns) - set(metadata.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing modalities in {path.name}: {missing}")
        frame = pd.read_parquet(path, columns=columns)
        rows.append({"name": path.name, "rows": len(frame)})
        group_values = [trainer.get_functional_groups(s) for s in frame["smiles"]]
        canonical = []
        for s, groups in zip(frame["smiles"], group_values):
            canonical.append(None if groups is None else Chem.MolToSmiles(Chem.MolFromSmiles(str(s).strip().replace(" ", "")), canonical=True))
        for modality, (_, column, _, width) in MODALITIES.items():
            group_batch, identity_batch = [], []
            for row_index, (spectrum, groups, molecule) in enumerate(zip(frame[column], group_values, canonical)):
                if spectrum is None or groups is None or (np.isscalar(spectrum) and pd.isna(spectrum)):
                    audit[modality]["excluded_rows"] += 1
                    continue
                dense = make_msms_spectrum(spectrum) if modality.startswith("msms") else np.asarray(spectrum)
                length = len(dense)
                counts = audit[modality]["native_lengths"]
                counts[str(length)] = counts.get(str(length), 0) + 1
                if length != width:
                    raise ValueError(f"Unexpected native length {length} for {modality}; refusing implicit resampling")
                audit[modality]["nonfinite_values"] += int((~np.isfinite(dense)).sum())
                if number == 1 and len(group_batch) < 2:
                    expected = np.nan_to_num(np.asarray(dense, dtype=np.float32), nan=0, posinf=0, neginf=0)
                    processed = preprocess_spectra([dense], parse_preprocessing("none"), width)[0]
                    if not np.array_equal(expected, processed):
                        raise ValueError("Native 'none' representation check failed")
                group_batch.append(groups)
                identity_batch.append(molecule)
                sample_digests[modality].update(f"{path.name}:{row_index}:{molecule}\n".encode())
            labels[modality].append(np.asarray(group_batch, dtype=np.int32))
            identities[modality].extend(identity_batch)
    split_root = root / "shared_splits"
    split_root.mkdir(exist_ok=True)
    result = {"file_rows": rows, "modalities": {}, "grouped_by_molecule": False,
              "notes": ["Original split artifacts were not available in Git. Reconstructed artifacts are not proof of historical identity.",
                        "MS/MS uses int(m/z * 10), clipped to 0..9999; last peak in each bin overwrites earlier peaks.",
                        "none retains native width, uses float32 and replaces non-finite values with zero."]}
    generated_splits = {}
    for modality in MODALITIES:
        y = np.concatenate(labels[modality])
        if y.ndim != 2 or y.shape[1] != 37:
            raise ValueError("Expected exactly 37 functional groups")
        molecules = np.asarray(identities[modality])
        # Ensure the original split generator uses the same label implementation.
        import importlib
        loader = importlib.import_module(f"spectroscopy_qml.mps_classifier.{modality}.data_loader")
        expected_smarts = [Chem.MolToSmarts(v) for v in trainer.functional_groups.values()]
        observed_smarts = [Chem.MolToSmarts(v) for v in loader.FUNCTIONAL_GROUPS.values()]
        if expected_smarts != observed_smarts or list(loader.FUNCTIONAL_GROUPS) != list(trainer.functional_groups):
            raise ValueError("Split label definitions do not match XGBoost labels")
        item = {**audit[modality], "samples": len(y), "label_digest": label_digest(y),
                "sample_order_sha256": sample_digests[modality].hexdigest(), "splits": {}}
        for seed in SEEDS:
            reuse_key = (seed, item["label_digest"], item["sample_order_sha256"])
            target = split_root / f"{modality}_seed_{seed}.npz"
            original = repo / "benchmark/preprocessing_training/shared_splits_600" / target.name
            provenance_path = target.with_suffix(".provenance.json")
            if target.exists():
                provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {"kind": "existing_unproven", "historical_identity_verified": False}
            elif original.exists():
                publish(original, target)
                provenance = {"kind": "copied_original", "source": str(original), "historical_identity_verified": True}
            else:
                if not accept_reconstructed:
                    raise RuntimeError("Original splits missing; explicit acceptance of reconstructed splits required")
                reused = generated_splits.get(reuse_key)
                if reused is None:
                    local_path = local / target.name
                    create_split_artifact(y, local_path, seed)
                    publish(local_path, target)
                else:
                    publish(reused, target)
                provenance = {"kind": "deterministic_reconstruction", "historical_identity_verified": False,
                              "accepted_by_user": True, "algorithm": "unchanged create_split_artifact, seed and seed+1"}
                if reused is not None:
                    provenance["reused_identical_inputs_from"] = str(reused)
            split = load_split_artifact(target, y, expected_seed=seed)
            if not provenance.get("historical_identity_verified") and not accept_reconstructed:
                raise RuntimeError("Historical split identity is unverified")
            positive_counts = y[split["train"]].sum(axis=0)
            if ((positive_counts == 0) | (positive_counts == len(split["train"]))).any():
                raise ValueError(f"Constant training target prevents unchanged XGBoost fitting: {modality} {seed}")
            molecule_sets = {key: set(molecules[index]) for key, index in split.items()}
            overlaps = {f"{a}_{b}": len(molecule_sets[a] & molecule_sets[b])
                        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
            index_hashes = {key: hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest() for key, value in split.items()}
            record = {**provenance, "sha256": sha256(target), "counts": {k: len(v) for k,v in split.items()},
                      "label_digest": label_digest(y), "sample_order_sha256": item["sample_order_sha256"],
                      "index_sha256": index_hashes, "canonical_smiles_overlap": overlaps}
            write_json(provenance_path, record)
            item["splits"][str(seed)] = record
            if record.get("kind") == "deterministic_reconstruction":
                generated_splits.setdefault(reuse_key, target)
        result["modalities"][modality] = item
    write_json(root / "experiments/xgboost_raw_native/configurations/data_and_split_audit.json", result)
    return result


def backup(repo, root, local):
    backup_dir = root / "repository-backup" / (stamp() + "-" + git(repo, "rev-parse", "--short", "HEAD"))
    backup_dir.mkdir(parents=True)
    bundle = local / "repository.bundle"
    subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "bundle", "verify", str(bundle)], check=True)
    bundle_hash = publish(bundle, backup_dir / bundle.name)
    archive = local / "worktree.tar.gz"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz", "-o", str(archive), "HEAD"],
                   env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"), check=True)
    archive_hash = publish(archive, backup_dir / archive.name)
    write_json(backup_dir / "recovery.json", {
        "reference_commit": REFERENCE, "infrastructure_commit": git(repo, "rev-parse", "HEAD"),
        "branch": git(repo, "branch", "--show-current"), "bundle_sha256": bundle_hash,
        "worktree_sha256": archive_hash, "raw_data": str(root / "data/raw"),
        "restore": "GIT_LFS_SKIP_SMUDGE=1 git clone repository.bundle /root/spectroscopy-qml; then select the recorded infrastructure branch; restore requirements and runpod launcher. LFS bytes are separately preserved in data/raw with their original OID hashes.",
    })
    return str(backup_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=Path("/root/runpod-work"))
    parser.add_argument("--accept-reconstructed-splits", action="store_true")
    args = parser.parse_args()
    args.local.mkdir(parents=True, exist_ok=True)
    sources = check_science(args.repo)
    environment(args.root, args.local)
    backup_dir = backup(args.repo, args.root, args.local)
    data, manifest = dataset(args.repo, args.root)
    audit_and_splits(args.repo, args.root, args.local, data, args.accept_reconstructed_splits)
    ready = {"prepared": stamp(), "reference_commit": REFERENCE, "sources": sources,
             "dataset_manifest_sha256": sha256(manifest), "backup": backup_dir,
             "accept_reconstructed_splits": args.accept_reconstructed_splits,
             "audit_sha256": sha256(args.root / "experiments/xgboost_raw_native/configurations/data_and_split_audit.json")}
    write_json(args.root / "experiments/xgboost_raw_native/configurations/ready.json", ready)
    print("PREPARATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
