"""Typed experiment settings; data/output paths use the configuration project directory."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import math

TARGETS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "CoPx", "CoPz")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _unique_keys(pairs):
    """Do not silently replace an earlier setting in a hand-edited JSON file."""
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate configuration key: {key}")
        values[key] = value
    return values


@dataclass(frozen=True)
class ExperimentConfig:
    study: str = "km2kn"
    target: str = "Fy"
    side: str = "LL"
    data_format: str = "matlab"
    data_files: list[str] = field(
        default_factory=lambda: ["data/Data_Preprocessed_Walk.mat", "data/Data_Preprocessed_Run.mat"]
    )
    dataset_state: str = "legacy_normalized"
    target_units: str = "normalized_source_scale_unknown"
    normalization: str = "none"
    target_source: str = "force_plate"
    input_sides: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    allow_legacy_channel_order: bool = False
    model: str = "IMUNet"
    depth: int = 5
    width: int = 32
    kernel_size: int = 3
    cardinality: int = 5
    pooling: str = "mix"
    deep_supervision: bool = True
    window_length: int = 1024
    sample_rate: float = 256.0
    train_stride: int = 1024
    fold_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    seed: int = 2026
    validation_subjects: int = 1
    min_abs_pcc: float = 0.0
    max_pairwise_correlation: float = 1.0
    epochs: int = 500
    batch_size: int = 16
    learning_rate: float = 0.0003
    patience: int = 20
    lr_patience: int = 10
    loss: str = "mae"
    output_dir: str = "runs"
    project_dir: str | None = None

    def __post_init__(self):
        # Type checks precede arithmetic/membership: JSON true is an int in Python.
        choices = {
            "study": ("km2kn", "insole"),
            "target": TARGETS,
            "side": ("LL", "RL"),
            "data_format": ("matlab", "npz"),
            "dataset_state": ("source_native", "legacy_normalized"),
            "normalization": ("none", "train_minmax"),
            "target_source": ("force_plate", "insole"),
            "model": (
                "IMUNet",
                "MLMRSNet",
                "MLMRSNetV2",
                "LDNet",
                "UNet",
                "UNetPP",
                "UNet3P",
                "MultiResUNet",
                "FPN",
                "PSPNet",
            ),
            "pooling": ("avg", "max", "mix"),
            "loss": ("mae", "huber", "mse"),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}.")
        for name in ("data_files", "input_sides", "feature_names", "fold_ids"):
            values = getattr(self, name)
            if not isinstance(values, list):
                raise ValueError(f"{name} must be a JSON list.")
            if name == "fold_ids":
                if not values or any(type(v) is not int or not 1 <= v <= 5 for v in values):
                    raise ValueError("fold_ids must contain distinct integers from 1 to 5.")
            elif any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{name} must contain nonempty strings.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} cannot contain duplicates.")
        if any(side not in ("LL", "RL") for side in self.input_sides):
            raise ValueError("input_sides must contain LL and/or RL.")
        if self.study == "km2kn" and self.input_sides:
            raise ValueError("input_sides selects pressure inputs; leave it empty for Km2Kn.")
        for name in ("target_units", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.project_dir is not None and (
            not isinstance(self.project_dir, str) or not self.project_dir.strip()
        ):
            raise ValueError("project_dir must be a nonempty path string or null.")
        if not self.data_files or len(set(self.paths())) != len(self.data_files):
            raise ValueError("data_files must contain at least one distinct resolved path.")
        for name in ("deep_supervision", "allow_legacy_channel_order"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false, without quotes.")
        for name in (
            "depth",
            "width",
            "kernel_size",
            "window_length",
            "train_stride",
            "epochs",
            "batch_size",
            "validation_subjects",
            "patience",
            "lr_patience",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 - 5:
            raise ValueError("seed must be between 0 and 2**32 - 6.")
        if type(self.cardinality) is not int or self.cardinality < 0:
            raise ValueError("cardinality must be a nonnegative integer.")
        for name in ("learning_rate", "sample_rate", "min_abs_pcc", "max_pairwise_correlation"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if self.learning_rate <= 0 or self.sample_rate <= 0:
            raise ValueError("learning_rate and sample_rate must be positive.")
        if not 0 <= self.min_abs_pcc <= 1 or not 0 < self.max_pairwise_correlation <= 1:
            raise ValueError("Feature correlation thresholds are outside their valid ranges.")
        if self.depth >= self.window_length.bit_length() or self.window_length % (2**self.depth):
            raise ValueError("window_length must be divisible by 2**depth.")
        if self.model == "MLMRSNet" and self.cardinality:
            exponent = self.depth + self.cardinality - 1
            if exponent >= self.window_length.bit_length() or self.window_length % (2**exponent):
                raise ValueError("MLMRSNet window_length must align with its depth/cardinality scopes.")
        if self.model == "MultiResUNet" and self.width < 6:
            raise ValueError("MultiResUNet requires width >= 6 for its three filter branches.")
        if self.train_stride > self.window_length:
            raise ValueError("train_stride cannot exceed window_length.")
        if self.validation_subjects >= 7:
            raise ValueError("validation_subjects must leave a training participant in every outer fold.")
        if (
            self.dataset_state == "legacy_normalized"
            and self.target_units != "normalized_source_scale_unknown"
        ):
            raise ValueError("Legacy window-normalized targets cannot be assigned physical units.")
        if self.target_source == "insole" and self.target not in ("CoPx", "CoPz"):
            raise ValueError("The insole reference is supported only for CoP diagnostics.")

    @property
    def root(self):
        """Base for data and outputs, independent of the caller's working directory."""
        return Path(self.project_dir).expanduser().resolve() if self.project_dir else PROJECT_ROOT

    @classmethod
    def from_json(cls, path):
        from dataclasses import fields

        path = Path(path).expanduser().resolve()
        values = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)
        if not isinstance(values, dict):
            raise ValueError("An experiment configuration must be a JSON object.")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        root = path.parent.parent if path.parent.name == "configs" else path.parent
        if values.get("project_dir") is not None:
            if not isinstance(values["project_dir"], str) or not values["project_dir"].strip():
                raise ValueError("project_dir must be a nonempty path string or null.")
            root = root / Path(values["project_dir"]).expanduser()
        values["project_dir"] = str(root.resolve())
        return cls(**values)

    def to_dict(self):
        return {**asdict(self), "project_dir": str(self.root)}

    def paths(self):
        return [(self.root / Path(p).expanduser()).resolve() for p in self.data_files]
