"""Strict data contracts; never substitute another target source silently."""

from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np

IMU_CHANNELS = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "knee_adduction_r",
    "knee_rotation_r",
    "ankle_angle_r",
    "ankle_adduction_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "knee_adduction_l",
    "knee_rotation_l",
    "ankle_angle_l",
    "ankle_adduction_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
)
EXCLUDED_IMU = {"pelvis_tx", "pelvis_tz", "mtp_angle_r", "mtp_angle_l"}
FP_CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "CoPx", "CoPy", "CoPz")
INSOLE_CHANNELS = tuple(f"p{i}" for i in range(1, 17)) + (
    "acc1",
    "acc2",
    "acc3",
    "ang1",
    "ang2",
    "ang3",
    "Fy",
    "CoPx",
    "CoPz",
)


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    speed: np.ndarray
    mode: np.ndarray
    record: np.ndarray
    feature_names: tuple[str, ...]
    units: str
    source: str
    study: str | None = None
    target: str | None = None
    side: str | None = None
    sample_rate: float | None = None

    def __post_init__(self):
        for name in ("x", "y", "subject", "speed", "mode", "record"):
            if not isinstance(getattr(self, name), np.ndarray):
                raise ValueError(f"{name} must be a NumPy array.")
        for name in ("x", "y", "subject", "speed", "mode"):
            if getattr(self, name).dtype.kind not in "iuf":
                raise ValueError(f"{name} must contain real numeric values.")
        if self.x.ndim != 3 or self.y.ndim != 3 or self.y.shape[-1] != 1:
            raise ValueError("Expected X=(record,time,channel), Y=(record,time,1).")
        if self.x.shape[:2] != self.y.shape[:2] or min(self.x.shape) < 1:
            raise ValueError("Input and target records must be nonempty and sample-aligned.")
        if any(not isinstance(n, str) or not n.strip() for n in self.feature_names):
            raise ValueError("Feature names must be nonempty strings.")
        if len(self.feature_names) != self.x.shape[2] or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Every input channel needs a distinct name.")
        for name in ("subject", "speed", "mode", "record"):
            array = getattr(self, name)
            if array.ndim != 1 or len(array) != len(self.x):
                raise ValueError(f"{name} must contain one label per record.")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Nonfinite inputs/targets detected; repair the source instead of zero-filling.")
        if (
            not np.isfinite(self.subject).all()
            or np.any(self.subject < 0)
            or np.any(self.subject >= 2**63)
            or np.any(self.subject != np.floor(self.subject))
        ):
            raise ValueError("Participant IDs must be finite integers.")
        if (
            not np.isfinite(self.speed).all()
            or np.any(self.speed < 0)
            or not set(np.unique(self.mode)).issubset({0, 1})
        ):
            raise ValueError("Speed must be finite and nonnegative; mode must be 0=walk or 1=run.")
        if self.record.dtype.kind == "S":
            object.__setattr__(self, "record", np.char.decode(self.record, "utf-8"))
        if self.record.dtype.kind != "U" or any(not str(v).strip() for v in self.record):
            raise ValueError("Record identifiers must be nonempty strings.")
        if len(set(self.record.tolist())) != len(self.record):
            raise ValueError("Duplicate record identifiers; review duplicated source records.")
        if any(not isinstance(v, str) or not v.strip() for v in (self.units, self.source)):
            raise ValueError("Dataset units and reference source must be explicit strings.")
        if self.sample_rate is not None and (not np.isfinite(self.sample_rate) or self.sample_rate <= 0):
            raise ValueError("Dataset sample_rate must be positive and finite.")

    def take(self, indices):
        return replace(
            self,
            **{
                name: getattr(self, name)[indices]
                for name in ("x", "y", "subject", "speed", "mode", "record")
            },
        )


def _matlab_labels(handle, key):
    """Decode MATLAB v7.3 cell arrays of UTF-16 character vectors."""
    if key not in handle:
        return ()
    dataset = handle[key]
    if dataset.dtype.kind == "O":
        labels = []
        for ref in np.asarray(dataset).ravel():
            if not ref:
                raise ValueError(f"{key} contains an empty MATLAB string reference.")
            values = np.asarray(handle[ref]).ravel()
            if values.dtype.kind not in "ui" or np.any(values > 65535):
                raise ValueError(
                    f"{key}: export MATLAB strings as cellstr character vectors, not opaque string objects."
                )
            labels.append(values.astype("<u2").tobytes().decode("utf-16-le").rstrip("\x00"))
        return tuple(labels)
    if dataset.dtype.kind in "SU":
        return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(dataset).ravel())
    raise ValueError(f"Unsupported label encoding in {key}; supply a documented export.")


def _labels(handle, key, canonical, width, config):
    labels = _matlab_labels(handle, key)
    if not labels:
        if not config.allow_legacy_channel_order or width != len(canonical):
            raise ValueError(f"Missing {key}; verify channel order before enabling the legacy fallback.")
        labels = canonical
    if len(labels) != width or len(set(labels)) != width or any(not n.strip() for n in labels):
        raise ValueError(f"{key} does not match the stored channel dimension.")
    return labels


def _real_array(values, key, dtype=None):
    """Reject lossy type coercion: complex and text signals are not real measurements."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{key} must contain finite real numeric measurements.")
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(dtype)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} exceeds the finite {np.dtype(dtype)} range.")
    return values


def _mat_array(handle, key, dtype=np.float32):
    if key not in handle:
        raise ValueError(f"Missing MATLAB dataset {key}.")
    array = _real_array(handle[key], key, dtype)
    if array.ndim != 3:
        raise ValueError(f"{key}: expected MATLAB HDF5 (channel,time,record).")
    return array.transpose(2, 1, 0)


def _vector(array, key, count):
    """Accept MATLAB row/column vectors, but never flatten a metadata matrix."""
    values = np.asarray(array)
    if values.ndim not in (1, 2) or (values.ndim == 2 and 1 not in values.shape) or values.size != count:
        raise ValueError(f"{key} must be a vector with one value per record ({count}).")
    return values.reshape(-1)


def _scalar(value, key):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{key} metadata must be scalar.")
    result = value.item()
    return result.decode() if isinstance(result, bytes) else result


def _check_sample_rate(value, expected):
    rate = _scalar(value, "sample_rate")
    if (
        type(rate) not in (int, float)
        or not np.isfinite(rate)
        or rate <= 0
        or not np.isclose(rate, expected, rtol=1e-9, atol=0)
    ):
        raise ValueError("Configured sample_rate conflicts with numeric source sample-rate metadata.")


def load_matlab(path, config):
    import h5py

    with h5py.File(path, "r") as handle:
        if "sample_rate_hz" in handle.attrs:
            _check_sample_rate(handle.attrs["sample_rate_hz"], config.sample_rate)
        elif config.dataset_state == "source_native":
            raise ValueError("source_native exports must supply sample_rate_hz metadata.")
        if "normalization_state" in handle.attrs:
            state = _scalar(handle.attrs["normalization_state"], "normalization_state")
            if state != config.dataset_state:
                raise ValueError("Configured dataset_state conflicts with the MATLAB export metadata.")
        elif config.dataset_state != "legacy_normalized":
            raise ValueError("An export without scale provenance cannot be declared source_native.")
        if config.study == "km2kn":
            x = _mat_array(handle, "data_IMU")
            names = _labels(handle, "label_IMU", IMU_CHANNELS, x.shape[2], config)
            keep = [i for i, name in enumerate(names) if name not in EXCLUDED_IMU]
            x, names = x[:, :, keep], tuple(names[i] for i in keep)
        else:
            sides = config.input_sides or [config.side]
            inputs, names = [], []
            for side in sides:
                stream = _mat_array(handle, f"data_Insole_{side}")
                if stream.shape[2] < 16:
                    raise ValueError(
                        "The insole route requires the documented 16 pressure channels per foot."
                    )
                labels = _labels(handle, "label_Insole", INSOLE_CHANNELS, stream.shape[2], config)
                if tuple(labels[:16]) != INSOLE_CHANNELS[:16]:
                    raise ValueError("The first 16 insole channels must be verified p1 through p16.")
                inputs.append(stream[:, :, :16])
                names.extend(f"{side}_p{i}" for i in range(1, 17))
            x, names = np.concatenate(inputs, axis=2), tuple(names)
        target = _mat_array(handle, f"data_Force_FP_{config.side}", dtype=None)
        canonical = FP_CHANNELS[:6] if target.shape[2] == 6 else FP_CHANNELS
        labels = _labels(handle, "label_FP", canonical, target.shape[2], config)
        if config.target not in labels:
            raise ValueError(
                f"Target {config.target} absent from {config.target_source}; no substitute will be used."
            )
        y = target[:, :, [labels.index(config.target)]]
        if "subinfo" not in handle or "speedinfo" not in handle:
            raise ValueError(
                "Missing subinfo or speedinfo. The original MATLAB script omitted speedinfo; regenerate with the maintained exporter."
            )
        subject = _vector(handle["subinfo"], "subinfo", len(x))
        speed = _vector(handle["speedinfo"], "speedinfo", len(x))
        mode_name = Path(path).stem.lower()
        if "modeinfo" in handle:
            mode = _vector(handle["modeinfo"], "modeinfo", len(x))
            if ("walk" in mode_name) != ("run" in mode_name) and np.any(mode != int("run" in mode_name)):
                raise ValueError("modeinfo conflicts with the Walk/Run export filename.")
        elif ("walk" in mode_name) != ("run" in mode_name):
            mode = np.full(len(x), int("run" in mode_name))
        else:
            raise ValueError("Export must provide modeinfo or an unambiguous Walk/Run filename.")
        # Rows are treated as independent records. Never stitch pre-windowed rows together.
        if ("source_file" in handle) != ("window_start_sample" in handle):
            raise ValueError("Provide source_file and window_start_sample together.")
        if "source_file" in handle:
            sources = _matlab_labels(handle, "source_file")
            starts = _vector(handle["window_start_sample"], "window_start_sample", len(x))
            if (
                starts.dtype.kind not in "iuf"
                or not np.isfinite(starts).all()
                or np.any(starts < 0)
                or np.any(starts != np.floor(starts))
            ):
                raise ValueError("window_start_sample must contain nonnegative integer offsets.")
            if len(sources) != len(x) or len(starts) != len(x):
                raise ValueError("Source/interval provenance does not match the record count.")
            record = np.array(
                [
                    f"{source}@source_sample_{int(start)}"
                    for source, start in zip(sources, starts, strict=True)
                ]
            )
        else:
            record = np.array([f"{Path(path).name}:{i}" for i in range(len(x))])
        return Dataset(
            x,
            y,
            subject,
            speed,
            mode,
            record,
            names,
            config.target_units,
            config.target_source,
            config.study,
            config.target,
            config.side,
            config.sample_rate,
        )


def load_npz(path, config):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive with named arrays, not a single NPY array.")
    with archive as data:
        if "sample_rate" not in data:
            raise ValueError("NPZ sample-rate metadata is missing.")
        _check_sample_rate(data["sample_rate"], config.sample_rate)
        for key, expected in (
            ("study", config.study),
            ("target", config.target),
            ("side", config.side),
            ("dataset_state", config.dataset_state),
            ("target_units", config.target_units),
            ("target_source", config.target_source),
        ):
            if key not in data or str(_scalar(data[key], key)) != expected:
                raise ValueError(f"NPZ metadata {key} is missing or conflicts with the experiment.")
        required = {"X", "Y", "subject", "speed", "mode", "record", "feature_names"}
        if missing := required - set(data.files):
            raise ValueError(f"NPZ arrays missing: {sorted(missing)}")
        x, y = _real_array(data["X"], "X", np.float32), _real_array(data["Y"], "Y")
        if x.ndim != 3:
            raise ValueError("NPZ X must have (record,time,channel) dimensions.")
        names = _vector(data["feature_names"], "feature_names", x.shape[2])
        if names.dtype.kind not in "US":
            raise ValueError("NPZ feature_names must contain strings.")
        names = tuple(names.astype(str))
        if config.study == "insole":
            expected = tuple(
                f"{side}_p{i}" for side in (config.input_sides or [config.side]) for i in range(1, 17)
            )
            if names != expected:
                raise ValueError("NPZ pressure channels must match configured feet in p1-p16 order.")
        return Dataset(
            x,
            y,
            _vector(data["subject"], "subject", len(x)),
            _vector(data["speed"], "speed", len(x)),
            _vector(data["mode"], "mode", len(x)),
            _vector(data["record"], "record", len(x)),
            names,
            config.target_units,
            config.target_source,
            config.study,
            config.target,
            config.side,
            config.sample_rate,
        )


def load_dataset(config):
    parts = []
    for path in config.paths():
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. See docs/DATA_CONTRACT.md; use the demo command for synthetic validation."
            )
        parts.append((load_matlab if config.data_format == "matlab" else load_npz)(path, config))
    first = parts[0]
    for part in parts[1:]:
        if part.feature_names != first.feature_names or part.x.shape[1:] != first.x.shape[1:]:
            raise ValueError("Input files disagree on feature order or record dimensions.")
    merged = (
        first
        if len(parts) == 1
        else replace(
            first,
            **{
                name: np.concatenate([getattr(p, name) for p in parts], axis=0)
                for name in ("x", "y", "subject", "speed", "mode", "record")
            },
        )
    )
    if len(set(merged.record.tolist())) != len(merged.record):
        raise ValueError("Duplicate record identifiers across input files; review duplicated source records.")
    if config.study == "km2kn":
        keep = [i for i, name in enumerate(merged.feature_names) if name not in EXCLUDED_IMU]
        merged = replace(
            merged, x=merged.x[:, :, keep], feature_names=tuple(merged.feature_names[i] for i in keep)
        )
    if config.feature_names:
        missing = set(config.feature_names) - set(merged.feature_names)
        if missing:
            raise ValueError(f"Requested features unavailable: {sorted(missing)}")
        indices = [merged.feature_names.index(name) for name in config.feature_names]
        merged = replace(merged, x=merged.x[:, :, indices], feature_names=tuple(config.feature_names))
    return merged


def synthetic_dataset(config, records_per_subject=2):
    """Small deterministic signals for software checks, never research evidence."""
    rng = np.random.default_rng(config.seed)
    subject = np.repeat(np.arange(4, 13), records_per_subject)
    length = config.window_length * 2
    t = np.arange(length, dtype=np.float32) / config.sample_rate
    phase = rng.uniform(0, 2 * np.pi, (len(subject), 1))
    wave = np.sin(2 * np.pi * t[None, :] + phase)
    x = np.stack([wave, np.cos(2 * np.pi * t[None, :] + phase), wave**2], axis=2).astype(np.float32)
    y = (0.3 * x[:, :, 0:1] + 0.1 * x[:, :, 1:2] + 0.5).astype(np.float32)
    return Dataset(
        x,
        y,
        subject,
        np.full(len(x), 3.6),
        np.zeros(len(x), dtype=int),
        np.array([f"synthetic-{i}" for i in range(len(x))]),
        ("synthetic_sine", "synthetic_cosine", "synthetic_square"),
        "synthetic_arbitrary_units",
        "synthetic_only",
        config.study,
        config.target,
        config.side,
        config.sample_rate,
    )
