"""XGBoost's native external-memory histogram algorithm, with raw batches.

The iterator changes storage only. Every original feature and occupancy flag is
passed through RawCohort.features. Quantile cuts belong to the fitted tree
algorithm and use training data only, as in ordinary histogram XGBoost.
"""

from pathlib import Path

import xgboost as xgb


class NativeDataIter(xgb.DataIter):
    def __init__(
        self, cohort, modality, indices, prefix: Path, device, batch_size=4096, on_host=True
    ):
        self.cohort, self.modality, self.indices = cohort, modality, indices
        self.device, self.batch_size, self.position = device, batch_size, 0
        self.x, self.y = None, None
        if str(device).startswith("cuda"):
            try:
                import cupy

                self.cupy = cupy
            except ImportError as error:
                raise RuntimeError(
                    "GPU external-memory XGBoost requires the CuPy wheel matching your CUDA driver (e.g. cupy-cuda12x); use --device cpu or install it"
                ) from error
        super().__init__(cache_prefix=str(prefix), release_data=True, on_host=on_host)

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position >= len(self.indices):
            return False
        ids = self.indices[self.position : self.position + self.batch_size]
        self.x = self.cohort.features(self.modality, ids).reshape(len(ids), -1)
        # Initial placeholder target; the common trainer sets each label explicitly.
        self.y = self.cohort.labels[ids, 0].astype("float32")
        if str(self.device).startswith("cuda"):
            self.x, self.y = self.cupy.asarray(self.x), self.cupy.asarray(self.y)
        input_data(data=self.x, label=self.y)
        self.position += len(ids)
        return True


def external_matrices(
    cohort, modality, split, output, device, threads, batch_size=4096, on_host=True
):
    train_iter = NativeDataIter(
        cohort, modality, split["train"], output / "train_cache", device, batch_size, on_host
    )
    val_iter = NativeDataIter(
        cohort,
        modality,
        split["selection"],
        output / "selection_cache",
        device,
        batch_size,
        on_host,
    )
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=256, nthread=threads)
    dval = xgb.ExtMemQuantileDMatrix(val_iter, max_bin=256, nthread=threads, ref=dtrain)
    return dtrain, dval
