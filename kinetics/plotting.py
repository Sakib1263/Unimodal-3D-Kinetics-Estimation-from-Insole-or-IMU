"""Portable waveform plots; no TeX, display server, or data access on import."""

from pathlib import Path
import numpy as np


def plot_predictions(path, output=None, sample=0):
    import matplotlib.pyplot as plt

    with np.load(path, allow_pickle=False) as data:
        required = {"Y_true", "Y_pred", "sample_rate", "target", "units", "subject"}
        if missing := required - set(data.files):
            raise ValueError(f"Prediction archive missing: {sorted(missing)}")
        truth, prediction = data["Y_true"], data["Y_pred"]
        if (
            truth.shape != prediction.shape
            or truth.ndim != 3
            or truth.shape[-1] != 1
            or type(sample) is not int
            or not 0 <= sample < len(truth)
            or not np.isfinite(truth).all()
            or not np.isfinite(prediction).all()
        ):
            raise ValueError("Prediction archive must contain aligned NTC arrays and a valid sample index.")
        if any(data[key].ndim != 0 for key in ("sample_rate", "target", "units")):
            raise ValueError("Prediction sample_rate, target and units must be scalar metadata.")
        sample_rate = float(data["sample_rate"].item())
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("Prediction sample_rate must be positive and finite.")
        if data["subject"].shape != (len(truth),):
            raise ValueError("Prediction participant labels do not match the window count.")
        target, units = str(data["target"].item()), str(data["units"].item())
        source = str(data["target_source"].item()) if "target_source" in data else "reference unspecified"
        side = str(data["side"].item()) if "side" in data else ""
        title = f"{side} {target} | participant {data['subject'][sample]} | window {sample} | {source}"
        if source == "synthetic_only":
            title = "SYNTHETIC DEMO - " + title
        times = np.arange(truth.shape[1]) / sample_rate
        figure, axes = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        axes.plot(times, truth[sample, :, 0], label="Reference", color="#1f4e79")
        axes.plot(times, prediction[sample, :, 0], label="Prediction", color="#bc4b28", linestyle="--")
        axes.set(
            xlabel="Time (s)",
            ylabel=f"{target} ({units})",
            title=title,
        )
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(frameon=False)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
    return figure


def lowpass_then_resample(values, sample_rate, output_length=1024, cutoff=15.0):
    """Filter in the original time base, then normalize the stance to a fixed grid."""
    from scipy.signal import butter, sosfiltfilt, resample

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 16:
        raise ValueError("A finite one-dimensional signal of at least 16 samples is required.")
    if (
        not np.isfinite(sample_rate)
        or not np.isfinite(cutoff)
        or not 0 < cutoff < sample_rate / 2
        or type(output_length) is not int
        or output_length < 2
    ):
        raise ValueError("Require 0 < cutoff < Nyquist and output_length >= 2.")
    sos = butter(4, cutoff, fs=sample_rate, output="sos")
    return resample(sosfiltfilt(sos, values), output_length)


def stance_intervals(vertical_force, threshold, minimum_samples=1):
    """Half-open intervals, including a contact touching either recording boundary."""
    values = np.asarray(vertical_force)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Vertical force must be a finite one-dimensional signal.")
    if not np.isfinite(threshold) or type(minimum_samples) is not int or minimum_samples < 1:
        raise ValueError("Contact threshold must be finite; minimum_samples must be a positive integer.")
    active = np.r_[False, values > threshold, False].astype(int)
    starts, stops = np.flatnonzero(np.diff(active) == 1), np.flatnonzero(np.diff(active) == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True) if b - a >= minimum_samples]


def plot_cli(argv=None):
    import argparse
    import sys

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Plot a reference/prediction archive from a maintained run.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args(argv)
    figure = plot_predictions(args.predictions, args.output, args.sample)
    import matplotlib.pyplot as plt

    plt.close(figure)


FORCE_COLUMNS = ("RFx", "RFy", "RFz", "RMx", "RMy", "RMz", "LFx", "LFy", "LFz", "LMx", "LMy", "LMz")
INSOLE_COLUMNS = tuple(
    name
    for side in ("L", "R")
    for name in (
        *(f"{side}P{i}" for i in range(16)),
        f"{side}AX",
        f"{side}AY",
        f"{side}AZ",
        f"{side}GX",
        f"{side}GY",
        f"{side}GZ",
        f"{side}TF",
        f"{side}CX",
        f"{side}CY",
    )
)


def raw_stance_means(
    directory,
    kind,
    column,
    vertical_column,
    sample_rate,
    threshold,
    header=False,
    minimum_seconds=0.1,
    cutoff=15.0,
):
    """Average stances per participant/trial; never retain only the last file in a loop."""
    if kind not in ("force", "insole"):
        raise ValueError("kind must be force or insole.")
    if (
        not np.isfinite(sample_rate)
        or sample_rate <= 0
        or not np.isfinite(minimum_seconds)
        or minimum_seconds <= 0
    ):
        raise ValueError("sample_rate and minimum_seconds must be positive and finite.")
    if not np.isfinite(threshold) or not np.isfinite(cutoff) or not 0 < cutoff < sample_rate / 2:
        raise ValueError("Require a finite threshold and 0 < cutoff < Nyquist.")
    columns = FORCE_COLUMNS if kind == "force" else INSOLE_COLUMNS
    if column not in columns or vertical_column not in columns:
        raise ValueError("Requested signal/contact reference is absent from the selected CSV schema.")
    if column[0] != vertical_column[0]:
        raise ValueError("The plotted signal and contact reference must come from the same foot.")
    minimum_samples = max(16, int(np.ceil(minimum_seconds * sample_rate)))
    result = {}
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {directory}")
    for path in files:
        participant, separator, trial = path.stem.partition("_")
        if not separator or not participant or not trial:
            raise ValueError(f"Filename must identify participant and trial: {path.name}")
        if header:
            import csv

            with path.open(newline="", encoding="utf-8-sig") as stream:
                actual = next(csv.reader(stream), [])
            if tuple(name.strip() for name in actual) != columns:
                raise ValueError(f"CSV header/order does not match the documented {kind} schema: {path.name}")
        data = np.loadtxt(path, delimiter=",", skiprows=int(header), ndmin=2)
        if data.shape[1] != len(columns) or not np.isfinite(data).all():
            raise ValueError(f"Invalid {kind} CSV dimensions/values: {path.name}")
        intervals = stance_intervals(data[:, columns.index(vertical_column)], threshold, minimum_samples)
        segments = [
            lowpass_then_resample(data[start:stop, columns.index(column)], sample_rate, 101, cutoff)
            for start, stop in intervals
        ]
        if segments:
            result.setdefault(trial, {}).setdefault(participant, []).extend(segments)
    if not result:
        raise ValueError(
            "No qualifying contacts; review the reference signal, units, threshold, and duration."
        )
    return {
        trial: np.stack([np.mean(segments, axis=0) for _, segments in sorted(participants.items())])
        for trial, participants in sorted(result.items())
    }


def raw_plot_cli(argv=None):
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Participant-balanced stance plots from source CSVs.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kind", choices=("force", "insole"), required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--vertical-column", required=True)
    parser.add_argument("--sample-rate", required=True, type=float)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--minimum-seconds", type=float, default=0.1)
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--units", required=True, help="Verified source units of the plotted signal.")
    parser.add_argument("--header", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    groups = raw_stance_means(
        args.directory,
        args.kind,
        args.column,
        args.vertical_column,
        args.sample_rate,
        args.threshold,
        args.header,
        args.minimum_seconds,
        args.cutoff,
    )
    figure, axes = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for trial, participants in groups.items():
        mean = participants.mean(axis=0)
        phase = np.linspace(0, 100, mean.size)
        axes.plot(phase, mean, label=f"{trial} (n={len(participants)})")
        if len(participants) > 1:
            sd = participants.std(axis=0, ddof=1)
            axes.fill_between(phase, mean - sd, mean + sd, alpha=0.15)
    axes.set(
        xlabel="Stance (%)",
        ylabel=f"{args.column} ({args.units})",
        title="Participant means; shaded sample SD where n > 1",
        xlim=(0, 100),
    )
    axes.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
