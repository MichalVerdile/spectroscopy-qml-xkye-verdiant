# Paper learning-curve package (PRELIMINARY)

Preprocessing: FIR + Gaussian smoothing + sqrt transformation + SNV normalization. Input dimension: 600.

- `01_main_native_axes`: paper figures using native epochs or boosting steps; mean ± one sample standard deviation over seeds 41–45.
- `02_validation_vs_training_time`: best validation F1 versus pure model-fit time; both axes show mean ± sample standard deviation.
- `03_result_table`: final test Micro-F1, Macro-F1, and pure model-fit time as mean ± sample standard deviation, in CSV, Markdown, and LaTeX.
- `04_appendix_individual_seeds`: individual native-axis curves for every seed.
- `05_supplement_normalized_qualitative`: normalized 0–100% overlays, explicitly marked qualitative.

Test data are not used for learning curves. The test set is represented only in the final result table.

Current status: **PRELIMINARY**. Any incomplete XGBoost source is visibly marked `PRELIMINARY`/`in progress`; rerun this generator after the remaining jobs finish to produce the final package.
