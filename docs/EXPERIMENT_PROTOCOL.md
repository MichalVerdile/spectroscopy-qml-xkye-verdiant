# Prospective raw-native experiment protocol

Protocol version: `raw_eval_v1`. Freeze this protocol, data manifest, code commit,
search schedule and resource ceilings before opening final test predictions.
The exploratory smoke/pilot runs described in RESEARCH_FINDINGS are engineering
checks and cannot be used to choose a favorable scientific conclusion.

## Hypotheses, endpoints and selection

The primary endpoint is test micro F1 under validation-calibrated **global**
thresholds. For each of C-NMR, H-NMR, IR, MS/MS+ and MS/MS−, compare MPS−CNN,
MPS−XGBoost, TTN−CNN and TTN−XGBoost on exactly paired splits. These are 20
declared primary comparisons. The direction of interest is positive, but the
descriptive sign-flip calculation is two-sided and not a superiority guarantee.

Macro F1, weighted F1, micro/macro precision and recall, per-class F1, Hamming
loss, Brier score and binary cross-entropy are secondary. Fixed .5 thresholds
are a control; per-class calibrated thresholds are a secondary policy that
optimizes individual class F1, not micro F1. Do not choose the reporting policy
from test scores. State explicitly if macro improves while micro does not.

For a best-classical versus best-tensor comparison, select CNN/XGBoost and
MPS/TTN **using selection-partition micro F1 at .5**, independently for each seed
and modality. Earliest declared family wins a tie. Test scores never decide
which family is called best. Aggregation records selected families and paired
micro/macro differences as a secondary analysis.

## Shared cohort and splits

Validate all five spectra and the molecule together. Default invalid data policy
is a hard error; a predeclared `exclude` policy removes the whole row and records
why. Content hashes, original lengths and all 37 exact SMARTS definitions are
part of identity. A label-blind dataset-wide maximum fixes MS/MS padding width;
this is disclosed transductive shape information, not training of a signal
transform. Canonical model inputs and masks are identical for all families.

Seeds are 41, 42, 43, 44 and 45. Allocate approximately 80% training, 5%
checkpoint/hyperparameter selection, 5% threshold calibration and 10% final
test. Molecule groups use conservative non-isomeric canonical SMILES, preventing
stereoisomers from crossing partitions. This does not additionally standardize
tautomers, protonation states or salts; residual chemistry similarity is a
limitation. Label-aware group assignment approximates stratification; it does
not split an indivisible molecule group merely to match percentages.

`split.json` stores exact indices and sample IDs, SHA-256, each partition's
size, label supports and maximum absolute prevalence difference from the full
cohort. Inspect this quality report, especially missing rare-label positives,
before fitting; any revised split policy creates a new frozen experiment.

Run a separately labelled structure-generalization study with `--split-kind
scaffold`. Ring systems use Murcko scaffolds; acyclic molecules use generic
whole-molecule connectivity instead of collapsing all acyclic compounds into
one empty scaffold. Molecule isolation is checked again. Scaffold grouping is
a different evaluation population and must not be pooled with molecule splits.

## Training, search and budgets

All iterative models select the earliest checkpoint attaining the best selection
micro F1 at threshold .5. Patience 15 means stop after 15 consecutive evaluation
checks without strict improvement. Neural checks occur each epoch; an XGBoost
check follows one joint round across all nonconstant binary label models.
Individual trees do not choose independent checkpoints from their own log loss.
Constant training targets use constant predictors derived only from training.

AdamW, BCE, cosine schedule, global gradient clipping at 1 and no class weighting
are common neural settings. Epochs and boosting rounds differ in work and must
never be interpreted as equal compute simply because patience is equal.

The proposed B schedule is 12 distinct configurations per family/seed/modality,
including the declared default, followed by deterministic search with search
seed 2026. Search values live in `raw_experiments.py`. JSON overrides stay fixed
while unfixed coordinates are searched. CNN searches width, dropout, learning
rate and weight decay; XGBoost searches depth, learning rate, child weight,
row/column sampling, alpha/lambda and finite round limits; both TNs search their
declared dimensions, windows and optimization parameters. Architecture factors
outside that schedule belong to the separately budgeted C study.

The reference command uses at most 100 neural epochs, 200/400/800 searched XGB
rounds (default 300), 15-check patience, 1,200 seconds per trial, batch 64 and a
500-million neural parameter ceiling. The time cap is soft, checked at complete
epoch/round boundaries; it is not a scheduler wall-time guarantee. XGBoost
feature-matrix preparation is counted. One first complete selection check is
needed even when preparation is expensive. Memory/unsupported configuration
failures are reported, never silently discarded and replaced with favorable
trials. Review feasibility before running the frozen full experiment.

Equal trial count is one transparent fairness definition, not a proof of equal
search difficulty. Report the full schedule, failures, total recorded fitting
time, invocation wall/CPU time and Slurm resource accounting. CPU/GPU placement,
thread count and caps must be common. Hist/quantile construction is internal
tree learning on training features, not external spectral preprocessing.

## Calibration and final test access

`--phase tune` uses only training and selection for model optimization. After
configuration selection, `--phase final` calibrates the winning model using the
separate calibration partition. The global grid is .10 through .90 in .05
increments. Ties prefer .5, then the lower value. Fewer than 100 calibration
samples falls back to .5. Per-class tuning requires at least 20 positives and
20 negatives for that class; otherwise use the global threshold. Per-class
ties prefer the global value, then the lower value.

Persist thresholds, selection hashes and a test seal before predicting the test
partition. Export test and calibration probabilities, truth, sample IDs and
training label supports. The centralized evaluator computes every metric and
reliability bins. Reusing artifacts requires matching data, split, seed, config,
environment, code and protocol, plus file-integrity checks. Rerunning identical
code is allowed; changing configuration after test access requires a new
experiment and an honest account of that exploratory history.

## Experimental matrix and recommended order

| Stage | Purpose | Permitted selection |
|---|---|---|
| Engineering checks | Full-dimensional synthetic smoke, real shard integrity, small real pilot, GPU/Windows validation | No predictive claims |
| A: strict raw architecture controls | Feasible historical CNN/MPS adapters and XGBoost, same new protocol | Selection partition only; explicitly incomplete historical comparison |
| B: optimized raw models | All four native families, 5 modalities, 5 seeds, equal declared search | Selection partition; then independent calibration and sealed test |
| C: TN one-factor ablations | Dimensions, mapping, direction/scale, merge and normalization | Validation initially; independently held-out final confirmation |
| Structure generalization | Repeat frozen primary comparison using scaffold split | Same rules, separate result ID |
| Later preprocessing study | Explicit FIR/Gaussian/sqrt/SNV experiments | Separate data mode, architecture labels and leaderboard |

Original TTN's fixed Lorentz filters make it ineligible for strict A. Native CNN
can require hundreds of millions of parameters, and historical MPS may require
divisible flattened input lengths. Report infeasible A cells explicitly; B is
the complete four-family raw comparison. Do not downsample to fill A cells.

## Aggregation and interpretation

`aggregate_raw_native.py` normally requires all four models and five seeds for
each included modality. It rejects incompatible datasets, splits within a pair,
protocols, package/code identities, synthetic status and unequal trial counts.
`--allow-incomplete` is only for diagnostic partial summaries. Exported JSON/CSV
include means, sample SDs, paired differences, per-class support/F1, labels with
training support 1..99 and labels unseen in training. Label rarity is never
chosen from test support.

The 95% t intervals describe variation across the five split/training seeds on
one finite dataset. Their nominal coverage is not guaranteed: test cohorts
overlap and seeds are not independent population replications. Exact sign-flip
calculations have the same dependence caveat. With five nonzero pairs the
smallest two-sided p is .0625 even before Holm correction across the 20 primary
comparisons. These values cannot establish conventional .05 significance.
Holm-adjusted values are reported with that limitation, not as validated tests
of population superiority. Use more independent evidence, external datasets or
a prospectively designed group-level uncertainty analysis for stronger claims.

Parameter counts are meaningful within neural models; XGBoost reports tree
nodes separately. Serialized inference-model bytes provide a cross-family
storage measure, but include format overhead. Training checkpoints/optimizer
states are excluded from model bytes. Plot both F1 versus model bytes and,
for neural models, F1 versus parameter count. A ratio of F1 per parameter is
descriptive, not a causal efficiency or quality measure. GPU process memory is
sampled separately from Torch's allocator peak; unavailable values remain null.
