"""Reuse a saved fold on source-scale windows without refitting preprocessing."""

import json
from pathlib import Path
import numpy as np

from .config import ExperimentConfig
from .data import _check_sample_rate, _real_array
from .preprocessing import MinMaxTransform
from .runner import predict_batched, sha256_file


class FoldPredictor:
    """Load ``run/fold_XX`` and predict one target/foot on aligned NTC windows.

    Inputs use the training source's units and coordinate convention. Supply
    channel names and sample rate explicitly; references and participant labels
    are not needed. Predictions are returned in the saved source target units.
    """

    def __init__(self, fold_dir):
        import tensorflow as tf
        from models import layers  # noqa: F401 -- registers reusable custom layers

        fold_dir = Path(fold_dir)
        self.config = ExperimentConfig.from_json(fold_dir.parent / "config.json")
        split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
        features = split.get("features")
        if (
            not isinstance(features, list)
            or not features
            or any(not isinstance(n, str) or not n.strip() for n in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("Saved fold requires a distinct, nonempty feature-name list.")
        self.features = tuple(features)
        self.x_transform = (
            MinMaxTransform.from_dict(split["x_transform"]) if split.get("x_transform") is not None else None
        )
        self.y_transform = (
            MinMaxTransform.from_dict(split["y_transform"]) if split.get("y_transform") is not None else None
        )
        for transform, width in ((self.x_transform, len(features)), (self.y_transform, 1)):
            if transform is not None and transform.minimum.shape[2] != width:
                raise ValueError("Saved transform dimensions conflict with the fold's features/target.")
        model_file = fold_dir / "model.keras"
        if split.get("model_sha256") is not None and sha256_file(model_file) != split["model_sha256"]:
            raise ValueError("Model checksum does not match this fold; restore its original model.keras.")
        self._model = tf.keras.models.load_model(model_file, compile=False)
        if (
            self._model.input_shape != (None, self.config.window_length, len(features))
            or not isinstance(self._model.output, dict)
            or "out" not in self._model.output
            or tuple(self._model.output["out"].shape[1:]) != (self.config.window_length, 1)
        ):
            raise ValueError("Saved model dimensions conflict with the fold configuration.")
        self.units, self.target, self.side = self.config.target_units, self.config.target, self.config.side

    def predict_windows(self, windows, feature_names, sample_rate, batch_size=None):
        """Select saved channels in order, apply training scales and return NTC predictions."""
        if not isinstance(feature_names, (list, tuple, np.ndarray)):
            raise ValueError("Supply a list of input channel names.")
        names = tuple(feature_names)
        if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
            raise ValueError("Input feature names must be distinct nonempty strings.")
        windows = _real_array(windows, "Inference windows")
        if (
            windows.ndim != 3
            or not windows.size
            or windows.shape[1] != self.config.window_length
            or windows.shape[2] != len(names)
        ):
            raise ValueError("Supply nonempty NTC windows with the saved length and one name per channel.")
        if missing := set(self.features) - set(names):
            raise ValueError(f"Missing saved input features: {sorted(missing)}")
        _check_sample_rate(sample_rate, self.config.sample_rate)
        selected = windows[:, :, [names.index(name) for name in self.features]]
        selected = (
            self.x_transform.transform(selected)
            if self.x_transform is not None
            else _real_array(selected, "Model inputs", np.float32)
        )
        prediction = predict_batched(
            self._model, selected, self.config.batch_size if batch_size is None else batch_size
        )
        if (
            prediction.shape != (len(windows), self.config.window_length, 1)
            or not np.isfinite(prediction).all()
        ):
            raise ValueError("Saved model returned invalid prediction dimensions or values.")
        return self.y_transform.inverse(prediction) if self.y_transform is not None else prediction
