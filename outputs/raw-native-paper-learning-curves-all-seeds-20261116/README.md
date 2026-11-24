# Raw/native paper learning-curve package

Input: native/raw spectral resolution. No smoothing, representation transform, or normalization. The shared train/validation/test splits and seeds 41–45 match the preprocessed comparison run.

Included models: CNN, MPS, TTN, and XGBoost. Main figures use native epochs or XGBoost boosting steps and show mean ± one sample standard deviation. Test data are used only in the result tables. Normalized 0–100% figures are qualitative only. The XGBoost learning curves come from the isolated external-GPU source directory; external timings are not mixed into the timing comparison. XGBoost timing remains pending until the matching Sanjose timing job finishes. Training time denotes model fitting only and excludes loading, preprocessing, final test evaluation, and output writing.
