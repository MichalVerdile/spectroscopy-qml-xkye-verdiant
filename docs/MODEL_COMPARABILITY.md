# Model comparability

The common entrypoint is `scripts/run_raw_native_benchmark.py` → `raw_runner`
→ `raw_native` / `raw_labels` → `raw_splits` → a versioned model and trainer
→ `raw_evaluation` → `raw_aggregate`. No new training path imports a historical
signal loader. `pack_native` is the only measurement-to-model conversion shared
by the neural dataset, ordinary XGBoost and streamed XGBoost.

## What is identical

| Property | CNN | XGBoost | MPS | TTN |
|---|---|---|---|---|
| Raw samples, molecule IDs and labels | Common cohort | Common cohort | Common cohort | Common cohort |
| 37 SMARTS labels and order | Same | Same | Same | Same |
| Native NMR/IR values | All | All | All | All |
| MS/MS m/z, intensity, order, occupancy | All | All | All | All |
| Canonical model input hash | Same per modality | Same | Same | Same |
| Input layout | N × L × C | Flattened N × (L·C) | N × L × C | N × L × C |
| Split IDs, group isolation and seeds | Same | Same | Same | Same |
| Checkpoint objective | Selection micro F1 at .5 | Same, jointly across 37 labels | Same | Same |
| Patience | 15 checks | 15 joint checks | 15 checks | 15 checks |
| Calibration partition/grid/fallbacks | Central | Central | Central | Central |
| Final metrics/probability export | Central | Central | Central | Central |
| External signal preprocessing/augmentation | None | None | None | None |

Input equivalence does not imply identical inductive bias or identical hidden
representations. All observations reach the first trainable transformation.
Subsequent learned compression/pooling is part of a classifier and is not
claimed to be mathematically invertible.

Dense axes are fixed across samples and retained as metadata; models use common
coordinate positions. MS/MS physical m/z is an explicit feature. Sequence
models see source peak order, which need not be m/z-sorted. There is no hidden
sorting, peak truncation, duplicate collapse, m/z rounding or fixed 600/1800
conversion. A fixed-width XGBoost peak representation is feasible for this
finite dataset; it becomes potentially expensive for unusually long peak lists.

## What remains different

| Difference | Methodological treatment |
|---|---|
| Neural epoch versus boosting round | A boosting round updates each nonconstant label once. Patience counts evaluation opportunities, not equal compute. |
| Trainable neural parameters versus tree nodes | Count separately; compare serialized model bytes across families. Do not equate one tree node with one neural parameter. |
| Optimizer | AdamW/BCE for neural models; logistic histogram boosting per label for XGBoost. No class weighting in the new primary benchmark. |
| Hidden-state normalization | CNN BatchNorm after learned convolutions; MPS/TTN learned features and internal state normalization. No normalization of the shared raw measurements. |
| Training limits | Same declared trial count and wall-time cap; architecture-specific finite epochs/rounds and actual resource usage recorded. |
| GPU memory | Torch allocator peak and, when available, sampled process memory via nvidia-smi. XGBoost is not measured using the Torch allocator. Null means unavailable, not zero. |
| Resume | Neural epoch-boundary exact state/RNG recovery; XGBoost incomplete-trial restart. Record this cost when comparing computational efficiency. |
| Internal histogram approximation | XGBoost hist/quantile cuts are fitted on training features; they are not external spectral resampling. |

For strict compute accounting, use histories, elapsed training seconds,
fit wall/CPU seconds and per-trial invocation time. Recovered trials may have
unrecorded time after their last durable checkpoint; invocation fields are not
a complete cluster billing ledger. Keep Slurm accounting for resource studies.
The aggregate primary training-time field measures the completed training
trajectory; it must not be presented as exact total cost including failed jobs.

## Original architectures and native controls

| Historical family | Architecture in the source branch | New native behavior |
|---|---|---|
| CNN | Keras Conv1D 31/62 channels, kernel 11, two max pools; Flatten; Dense 4927/2785/1574; dropout ≈.486; Adam; val-loss stopping | `cnn_raw_native_v1`: multi-kernel residual CNN with masked average/max readout. `cnn_jung_native_adapter_v1`: explicit PyTorch architecture port with native input and mask channels. |
| XGBoost | 37 binary classifiers, depth 6, LR .05, subsample/column sample .8, per-label log-loss stopping | `xgb_raw_native_v1`: same family, native full input, joint micro-F1 selection; baseline mode retains .8 sampling. |
| MPS | Dual-scale bidirectional chains; local LayerNorm/MLP; physical dimension 450, bond 128; all site states projected to two 128-wide embeddings; 256-wide head | `mps_raw_native_v1`: smaller dimensions, complete learned segments, near-identity cores, internal stable contractions and residual readout. Historical adapter available when native flattened dimensions divide the original site counts. |
| TTN | Fixed Lorentz kernels and analytical derivatives; positional bias; normalized segment states; per-level QR merge; linear head | `ttn_raw_native_v1` begins with a trainable local encoder and no fixed filters. Historical `ttn_lorentz_feature_map` is a separate, non-strict feature-map architecture. |

`--variant baseline` does not secretly replace an incompatible historical model
with an optimized one. Historical TTN is rejected under strict raw. Large CNN
and MPS models require an explicitly increased parameter limit. Native MS/MS
can violate historical MPS divisibility; that case fails explicitly. Original
TensorFlow and legacy modules remain untouched for published-run reproduction.

Consequently, Experiment A is a set of feasible architecture controls, **not a
complete equivalent four-family historical leaderboard**. The complete strict
raw four-family comparison uses Experiment B. This limitation is part of the
scientific result, not grounds for inserting hidden filters or downsampling.

## Input, split and result identity

Data identity includes SHA-256 of source contents (not just filenames/mtimes),
all stored values/masks, labels, row provenance, vocabulary, physical-axis
metadata, filtering policy and loader implementation. Sample IDs incorporate
source file/row and canonical molecule; grouping collapses stereochemistry
conservatively, so stereoisomers cannot leak through molecule splitting.

Split identity includes exact indices, seed, dataset hash and group method.
Run identity adds architecture/configuration, full search schedule, limits,
evaluation/calibration protocol, code commit/source hash and package/hardware
environment. Complete artifacts are individually hashed and checked on reuse.
Aggregation rejects mixed input/split/protocol/configuration budgets and
different trial counts. Synthetic and real runs cannot be merged.
