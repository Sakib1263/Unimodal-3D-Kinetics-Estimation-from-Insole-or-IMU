"""Participant splits, training-only transforms, and record-bounded windows."""

from dataclasses import replace
import numpy as np

PAPER_TEST_SUBJECTS = {1: (11, 12), 2: (9, 10), 3: (7, 8), 4: (5, 6), 5: (4,)}


def subject_split(subject, fold, seed=2026, validation_subjects=1):
    subject = np.asarray(subject)
    if (
        subject.ndim != 1
        or subject.dtype.kind not in "iuf"
        or not np.isfinite(subject).all()
        or np.any(subject != np.floor(subject))
    ):
        raise ValueError("Participant IDs must form a finite integer vector.")
    if type(fold) is not int or fold not in PAPER_TEST_SUBJECTS:
        raise ValueError("fold must be an integer from 1 to 5.")
    available = set(np.unique(subject).astype(int))
    expected = set(range(4, 13))
    if available != expected:
        raise ValueError(f"The documented five-fold schedule requires P04-P12; got {sorted(available)}.")
    test = set(PAPER_TEST_SUBJECTS[fold])
    candidates = sorted(available - test)
    if type(seed) is not int or not 0 <= seed < 2**32 - 5:
        raise ValueError("seed must be an integer between 0 and 2**32 - 6.")
    if type(validation_subjects) is not int or not 1 <= validation_subjects < len(candidates):
        raise ValueError("Leave at least one training and one validation participant.")
    rng = np.random.default_rng(seed + fold)
    validation = set(rng.permutation(candidates)[:validation_subjects].tolist())
    training = set(candidates) - validation
    return tuple(np.flatnonzero(np.isin(subject, sorted(ids))) for ids in (training, validation, test))


class MinMaxTransform:
    """A reversible channel-wise transform fitted only on training records."""

    def fit(self, x):
        x = np.asarray(x)
        if x.ndim != 3 or not x.size or x.dtype.kind not in "iuf" or not np.isfinite(x).all():
            raise ValueError("Fit requires a finite nonempty (record,time,channel) array.")
        x = np.asarray(x, dtype=np.float64)
        self.minimum = np.min(x, axis=(0, 1), keepdims=True)
        span = np.max(x, axis=(0, 1), keepdims=True) - self.minimum
        if not np.isfinite(span).all():
            raise ValueError("Training range overflowed; review source units.")
        self.constant_channels = np.flatnonzero(span.ravel() == 0).tolist()
        self.scale = np.where(span > 0, span, 1.0)
        return self

    def _validate_input(self, x):
        if not hasattr(self, "minimum"):
            raise ValueError("Fit or restore the training transform before applying it.")
        x = np.asarray(x)
        if (
            x.ndim != 3
            or not x.size
            or x.shape[2] != self.minimum.shape[2]
            or x.dtype.kind not in "iuf"
            or not np.isfinite(x).all()
        ):
            raise ValueError("Transform input must be finite NTC data with the fitted channel count.")
        return x

    def transform(self, x):
        x = self._validate_input(x)
        with np.errstate(over="ignore", invalid="ignore"):
            result = ((x - self.minimum) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Transformed inputs exceed finite float32 range; review the source scale.")
        return result

    def inverse(self, x):
        result = self._validate_input(x) * self.scale + self.minimum
        if not np.isfinite(result).all():
            raise ValueError("Inverse transform overflowed; review source units and predictions.")
        return result

    def to_dict(self):
        return {
            "minimum": self.minimum.ravel().tolist(),
            "scale": self.scale.ravel().tolist(),
            "fit_partition": "training_only",
            "clipping": False,
            "constant_channels": self.constant_channels,
        }

    @classmethod
    def from_dict(cls, values):
        """Restore a fold's recorded training transform for model reuse."""
        result = cls()
        if not isinstance(values, dict) or not {"minimum", "scale"} <= values.keys():
            raise ValueError("A serialized transform requires minimum and scale vectors.")
        if (
            values.get("clipping", False) is not False
            or values.get("fit_partition", "training_only") != "training_only"
        ):
            raise ValueError("Only unclipped training-fitted min-max transforms are supported.")
        minimum, scale = (np.asarray(values[key]) for key in ("minimum", "scale"))
        if any(a.ndim != 1 or a.dtype.kind not in "iuf" for a in (minimum, scale)):
            raise ValueError("Serialized minimum and scale must be real numeric vectors.")
        result.minimum = minimum.astype(np.float64).reshape(1, 1, -1)
        result.scale = scale.astype(np.float64).reshape(1, 1, -1)
        if (
            result.minimum.shape != result.scale.shape
            or not result.scale.size
            or not np.isfinite(result.minimum).all()
            or not np.isfinite(result.scale).all()
            or np.any(result.scale <= 0)
        ):
            raise ValueError("Invalid serialized training transform.")
        result.constant_channels = values.get("constant_channels", [])
        if (
            not isinstance(result.constant_channels, list)
            or any(type(i) is not int or not 0 <= i < minimum.size for i in result.constant_channels)
            or len(set(result.constant_channels)) != len(result.constant_channels)
        ):
            raise ValueError("constant_channels must contain distinct fitted channel indices.")
        return result


def select_training_features(x, y, min_abs_pcc=0.0, max_pairwise_correlation=1.0):
    """Greedy pairwise pruning; this is not a multivariate VIF estimator."""
    if min_abs_pcc == 0 and max_pairwise_correlation == 1:
        kept = np.flatnonzero(np.max(x, axis=(0, 1)) > np.min(x, axis=(0, 1))).tolist()
        if not kept:
            raise ValueError(
                "Feature selection retained no channels; inspect training signals and thresholds."
            )
        return kept
    # Accumulate centered covariance in bounded batches instead of copying every
    # flattened waveform into several float64 matrices.
    means = np.mean(x, axis=(0, 1), dtype=np.float64)
    target_mean = np.mean(y, dtype=np.float64)
    covariance = np.zeros((x.shape[-1], x.shape[-1]), dtype=np.float64)
    cross = np.zeros(x.shape[-1], dtype=np.float64)
    target_sum_squares = 0.0
    for start in range(0, len(x), 32):
        centered = np.asarray(x[start : start + 32], dtype=np.float64).reshape(-1, x.shape[-1]) - means
        target = np.asarray(y[start : start + 32], dtype=np.float64).reshape(-1) - target_mean
        covariance += centered.T @ centered
        cross += centered.T @ target
        target_sum_squares += np.dot(target, target)
    norms = np.sqrt(np.maximum(np.diag(covariance), 0))
    target_norm = np.sqrt(target_sum_squares)
    kept = []
    for i, norm in enumerate(norms):
        if norm == 0:
            continue
        if min_abs_pcc > 0:
            corr = 0.0 if target_norm == 0 else cross[i] / (norm * target_norm)
            if abs(corr) < min_abs_pcc:
                continue
        if max_pairwise_correlation < 1 and any(
            abs(covariance[i, j] / (norm * norms[j])) > max_pairwise_correlation for j in kept
        ):
            continue
        kept.append(i)
    if not kept:
        raise ValueError("Feature selection retained no channels; inspect training signals and thresholds.")
    return kept


def window_records(data, length, stride):
    """Extract aligned windows inside each row, without concatenating unrelated records."""
    if (
        type(length) is not int
        or type(stride) is not int
        or not 1 <= stride <= length
        or length > data.x.shape[1]
    ):
        raise ValueError("Require 1 <= stride <= length <= source record length.")
    starts = np.arange(0, data.x.shape[1] - length + 1, stride)
    # Record-major ordering preserves labels and makes interval provenance inspectable.
    x = np.stack([row[start : start + length] for row in data.x for start in starts])
    y = np.stack([row[start : start + length] for row in data.y for start in starts])
    return replace(
        data,
        x=x,
        y=y,
        subject=np.repeat(data.subject, len(starts)),
        speed=np.repeat(data.speed, len(starts)),
        mode=np.repeat(data.mode, len(starts)),
        record=np.array([f"{record}@{start}:{start + length}" for record in data.record for start in starts]),
    )


def prepare_fold(data, config, fold):
    train_idx, val_idx, test_idx = subject_split(data.subject, fold, config.seed, config.validation_subjects)
    train, val, test = (data.take(idx) for idx in (train_idx, val_idx, test_idx))
    # Use the *source-scale training range*, even when no transform is requested.
    # A constant training reference has no meaningful range-normalized error.
    target_range = float(np.ptp(np.asarray(train.y, dtype=np.float64)))
    chosen = select_training_features(train.x, train.y, config.min_abs_pcc, config.max_pairwise_correlation)
    selected_names = tuple(data.feature_names[i] for i in chosen)
    train, val, test = (
        replace(part, x=part.x[:, :, chosen], feature_names=selected_names) for part in (train, val, test)
    )
    x_transform = y_transform = None
    if config.normalization == "train_minmax":
        x_transform, y_transform = MinMaxTransform().fit(train.x), MinMaxTransform().fit(train.y)
        train, val = (
            replace(part, x=x_transform.transform(part.x), y=y_transform.transform(part.y))
            for part in (train, val)
        )
        # Test references are never optimization targets. Preserve their exact
        # source values instead of normalizing and then numerically inverting them.
        test = replace(test, x=x_transform.transform(test.x))
    train = window_records(train, config.window_length, config.train_stride)
    val = window_records(val, config.window_length, config.window_length)
    test = window_records(test, config.window_length, config.window_length)
    manifest = {
        "fold": fold,
        "subjects": {
            name: sorted(np.unique(part.subject).astype(int).tolist())
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "window_counts": {
            name: len(part.x) for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "features": list(selected_names),
        "feature_selection_fit": "training_only",
        "x_transform": x_transform.to_dict() if x_transform else None,
        "y_transform": y_transform.to_dict() if y_transform else None,
        "augmentation_available": data.x.shape[1] > config.window_length,
        "training_stride": config.train_stride,
        "test_target_scale": "source",
        "source_record_samples": data.x.shape[1],
        "discarded_tail_samples_per_record": {
            name: (data.x.shape[1] - config.window_length) % stride
            for name, stride in (
                ("train", config.train_stride),
                ("validation", config.window_length),
                ("test", config.window_length),
            )
        },
        "nrmse_denominator": target_range if target_range > 0 else None,
        "nrmse_definition": "source-scale RMSE / training-record target range; null if range is zero",
    }
    return train, val, test, y_transform, manifest
