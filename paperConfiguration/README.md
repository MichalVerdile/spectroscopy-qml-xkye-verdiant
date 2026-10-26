# Paper configuration benchmark

This directory keeps the frozen model configuration and its Slurm entrypoint
together. The benchmark extends the paper setup from IR and C-NMR to all five
available one-dimensional spectra without tuning the architectures on the new
modalities.

Files:

- `paper_model_configurations.yaml`: complete configuration snapshot.
- `run_all_models_all_spectra_seeds_41_45.sbatch`: Slurm job for all four
  models, all five spectra and seeds 41-45.

Submit from the repository root:

```bash
sbatch paperConfiguration/run_all_models_all_spectra_seeds_41_45.sbatch
```

The job writes results to `benchmark/paper_all_spectra_seed_41_45/`. Completed
results are validated and skipped after a requeue. CNN, MPS and TTN use early
stopping with patience 15. XGBoost remains unchanged.

The paper only defines MPS site layouts for IR and C-NMR. The H-NMR and MS/MS
layouts recorded in the YAML file are the repository's frozen extension to the
additional modalities, not newly tuned paper results.
