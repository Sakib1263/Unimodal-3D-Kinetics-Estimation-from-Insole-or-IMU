"""Signed waveform metrics with explicit undefined values and balanced summaries."""

import csv
from pathlib import Path
import numpy as np

from .artifacts import atomic_text

METRIC_NAMES = ("mae", "rmse", "nrmse", "pcc", "r2", "ccc")


def waveform_metrics(truth, prediction, normalization_scale=1.0):
    if any(np.asarray(a).dtype.kind not in "iuf" for a in (truth, prediction)):
        raise ValueError("Metrics require real numeric arrays.")
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("Truth and prediction must be nonempty and have equal shapes.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values; missing references must not become zeros.")
    truth, prediction = truth.ravel(), prediction.ravel()
    if normalization_scale is not None and (
        not np.isscalar(normalization_scale)
        or np.asarray(normalization_scale).dtype.kind not in "iuf"
        or not np.isfinite(normalization_scale)
        or normalization_scale <= 0
    ):
        raise ValueError("normalization_scale must be positive and finite, or None for an undefined range.")
    # A shared relative scale prevents squared sums from overflowing or
    # underflowing and turning a perfect correlation into zero or null.
    magnitude = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction)))) or 1.0
    truth, prediction = truth / magnitude, prediction / magnitude
    error = prediction - truth
    mse = np.mean(error**2)
    t, p = truth - truth.mean(), prediction - prediction.mean()
    tss, pss = np.dot(t, t), np.dot(p, p)
    denominator = truth.var() + prediction.var() + (truth.mean() - prediction.mean()) ** 2
    result = {
        "mae": float(np.mean(np.abs(error)) * magnitude),
        "rmse": float(np.sqrt(mse) * magnitude),
        "nrmse": float((np.sqrt(mse) * magnitude) / normalization_scale)
        if normalization_scale is not None
        else None,
        "pcc": float(np.clip(np.dot(t, p) / (np.sqrt(tss) * np.sqrt(pss)), -1, 1))
        if tss > 0 and pss > 0
        else None,
        "r2": float(1 - np.dot(error, error) / tss) if tss > 0 else None,
        "ccc": float(2 * np.mean(t * p) / denominator) if denominator > 0 else None,
    }
    if any(v is not None and not np.isfinite(v) for v in result.values()):
        raise ValueError("Metric arithmetic overflowed; review source units and numerical range.")
    return result


def evaluate_windows(data, predictions, fold, normalization_scale=1.0):
    if predictions.shape != data.y.shape:
        raise ValueError(
            "Prediction shape differs from the target. Select the named full-resolution out head."
        )
    rows = []
    for i in range(len(data.y)):
        rows.append(
            {
                "fold": fold,
                "subject": int(data.subject[i]),
                "speed": float(data.speed[i]),
                "mode": int(data.mode[i]),
                "record_window": str(data.record[i]),
                "units": data.units,
                **waveform_metrics(data.y[i], predictions[i], normalization_scale),
            }
        )
    return rows


def balanced_summary(rows):
    """Average windows within participant, then participants with equal weight."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation.")
    participants = []
    for subject in sorted({r["subject"] for r in rows}):
        current = [r for r in rows if r["subject"] == subject]
        row = {"subject": subject, "windows": len(current)}
        for metric in METRIC_NAMES:
            values = [r[metric] for r in current if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[f"{metric}_valid_windows"] = len(values)
        participants.append(row)
    summary = {
        "aggregation": "equal_participant_mean_of_window_metrics",
        "participants": len(participants),
        "windows": len(rows),
        "correlation_scale": "fraction, not percent",
    }
    for metric in METRIC_NAMES:
        values = [r[metric] for r in participants if r[metric] is not None]
        summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "valid_participants": len(values),
        }
    return participants, summary


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to export an empty result table.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("Every exported result row must have the same columns.")
    with atomic_text(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
