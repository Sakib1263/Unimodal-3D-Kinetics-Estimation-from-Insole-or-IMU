"""Fold-isolated training and provenance. No model or dataset is created on import."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import platform
import uuid
import numpy as np

from .config import PROJECT_ROOT
from .artifacts import atomic_text
from .data import load_dataset, synthetic_dataset
from .metrics import balanced_summary, evaluate_windows, write_csv
from .preprocessing import prepare_fold, subject_split


def write_json(path, value):
    with atomic_text(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(data):
    """Bind provenance to the arrays actually used, including in-memory inputs."""
    digest = hashlib.sha256()
    for name in ("x", "y", "subject", "speed", "mode", "record"):
        array = getattr(data, name)
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
    digest.update(json.dumps(data.feature_names).encode())
    return digest.hexdigest()


def provenance(config, data):
    packages = {}
    for name in ("tensorflow", "keras", "numpy", "scipy", "h5py", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, ["Git state unavailable"]
    files = sorted(
        [
            *PROJECT_ROOT.joinpath("kinetics").glob("*.py"),
            *PROJECT_ROOT.joinpath("models").glob("*.py"),
            PROJECT_ROOT / "Data_Preprocessing.m",
        ]
    )
    return {
        "git_commit": commit,
        "git_status": status,
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in files if p.is_file()},
        "input_sha256": {}
        if data.source == "synthetic_only"
        else {str(p): sha256_file(p) if p.is_file() else None for p in config.paths()},
        "loaded_dataset_sha256": dataset_fingerprint(data),
        "input_hash_note": "File hashes describe available exports; the loaded-array hash binds the actual inputs.",
        "dataset_shape": list(data.x.shape),
        "units": data.units,
        "target_source": data.source,
        "synthetic_only": data.source == "synthetic_only",
        "protocol": "maintained_subject_disjoint_validation_v1",
        "paper_reproduction_certified": False,
        "metric_domain": "stored source scale; inverse training transform when enabled",
        "nrmse_denominator": "source-scale training target range, recorded per fold; null if constant",
    }


def demo_configuration(config):
    return replace(
        config,
        model="UNet",
        depth=2,
        width=4,
        cardinality=2,
        kernel_size=3,
        window_length=64,
        sample_rate=64.0,
        train_stride=32,
        epochs=1,
        batch_size=8,
        fold_ids=[1],
        feature_names=[],
        normalization="train_minmax",
        dataset_state="source_native",
        target_units="synthetic_arbitrary_units",
    )


def inspect_dataset(config, data=None):
    data = load_dataset(config) if data is None else data
    for name in ("study", "target", "side", "sample_rate"):
        observed = getattr(data, name)
        if observed is not None and observed != getattr(config, name):
            raise ValueError(f"Loaded dataset {name} conflicts with the requested experiment.")
    if data.source != "synthetic_only" and (
        data.source != config.target_source or data.units != config.target_units
    ):
        raise ValueError("Loaded dataset reference source/units conflict with the requested experiment.")
    if data.x.shape[1] < config.window_length:
        raise ValueError("Source records are shorter than the requested window_length.")
    folds = []
    for fold in config.fold_ids:
        indices = subject_split(data.subject, fold, config.seed, config.validation_subjects)
        folds.append(
            {
                "fold": fold,
                "subjects": {
                    name: sorted(np.unique(data.subject[idx]).astype(int).tolist())
                    for name, idx in zip(("train", "validation", "test"), indices, strict=True)
                },
                "planned_window_counts": {
                    name: len(idx) * ((data.x.shape[1] - config.window_length) // stride + 1)
                    for name, idx, stride in zip(
                        ("train", "validation", "test"),
                        indices,
                        (config.train_stride, config.window_length, config.window_length),
                        strict=True,
                    )
                },
                "feature_selection_and_transforms": "fitted on training records during fold preparation",
            }
        )
    return {
        "input_shape": list(data.x.shape),
        "target_shape": list(data.y.shape),
        "participants": sorted(np.unique(data.subject).astype(int).tolist()),
        "features": list(data.feature_names),
        "units": data.units,
        "target_source": data.source,
        "augmentation_available": data.x.shape[1] > config.window_length,
        "folds": folds,
    }


def _tf_dataset(tf, x, targets, batch_size, shuffle=False, seed=2026):
    data = tf.data.Dataset.from_tensor_slices((x, targets))
    if shuffle:
        data = data.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return data.batch(batch_size).with_options(options).prefetch(1)


def predict_batched(model, x, batch_size=16):
    """Avoid assumptions about output order and avoid constructing a prediction dataset."""
    if type(batch_size) is not int or batch_size < 1 or len(x) == 0:
        raise ValueError("Prediction requires nonempty inputs and a positive integer batch_size.")
    return np.concatenate(
        [
            np.asarray(model(x[start : start + batch_size], training=False)["out"])
            for start in range(0, len(x), batch_size)
        ],
        axis=0,
    )


def run_experiment(config, data=None):
    data = load_dataset(config) if data is None else data
    inspect_dataset(config, data)  # Fail on invalid splits before creating a run.
    import tensorflow as tf
    from .modeling import build_model, compile_model, prepare_targets

    run_label = "demo" if data.source == "synthetic_only" else config.study
    run_id = f"{run_label}_{config.target}_{config.side}_{config.model}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    run_dir = config.root / config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "status.json", {"status": "running"})
    all_rows = []
    completed = []
    fold = None
    try:
        write_json(run_dir / "config.json", config.to_dict())
        write_json(run_dir / "provenance.json", provenance(config, data))
        for fold in config.fold_ids:
            write_json(
                run_dir / "status.json",
                {
                    "status": "running",
                    "current_fold": fold,
                    "folds_completed": completed,
                },
            )
            print(
                f"Fold {fold}: preparing independent training, validation, and test participants", flush=True
            )
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(config.seed + fold)
            train, val, test, inverse, manifest = prepare_fold(data, config, fold)
            fold_dir = run_dir / f"fold_{fold:02d}"
            fold_dir.mkdir()
            write_json(fold_dir / "split.json", manifest)
            model = compile_model(build_model(config, train.x.shape[2]), config)
            train_targets, val_targets = prepare_targets(model, train.y), prepare_targets(model, val.y)
            train_ds = _tf_dataset(tf, train.x, train_targets, config.batch_size, True, config.seed + fold)
            val_ds = _tf_dataset(tf, val.x, val_targets, config.batch_size)
            best_path = fold_dir / "best.weights.h5"
            callbacks = [
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=config.patience, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.1, patience=config.lr_patience
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    str(best_path), monitor="val_loss", save_best_only=True, save_weights_only=True
                ),
                tf.keras.callbacks.CSVLogger(str(fold_dir / "training.csv")),
            ]
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.epochs,
                callbacks=callbacks,
                shuffle=False,
                verbose=2,
            )
            if not best_path.exists() or any(
                not np.isfinite(v) for values in history.history.values() for v in values
            ):
                raise RuntimeError(
                    "Training produced nonfinite values or no valid checkpoint; test evaluation was not performed."
                )
            model.load_weights(best_path)
            model.save(fold_dir / "model.keras")
            manifest["model_sha256"] = sha256_file(fold_dir / "model.keras")
            write_json(fold_dir / "split.json", manifest)
            predictions = predict_batched(model, test.x, config.batch_size)
            normalization_scale = manifest["nrmse_denominator"]
            if inverse:
                predictions = inverse.inverse(predictions)
            rows = evaluate_windows(test, predictions, fold, normalization_scale)
            write_csv(fold_dir / "metrics_by_window.csv", rows)
            np.savez_compressed(
                fold_dir / "predictions.npz",
                Y_true=test.y,
                Y_pred=predictions,
                subject=test.subject,
                speed=test.speed,
                mode=test.mode,
                record=test.record,
                target=config.target,
                study=config.study,
                side=config.side,
                units=data.units,
                target_source=data.source,
                sample_rate=config.sample_rate,
                synthetic_only=data.source == "synthetic_only",
            )
            write_json(
                fold_dir / "history.json",
                {k: [float(v) for v in values] for k, values in history.history.items()},
            )
            all_rows.extend(rows)
            completed.append(fold)
            del model, train_ds, val_ds
        participants, summary = balanced_summary(all_rows)
        summary.update(
            units=data.units,
            target=config.target,
            target_source=data.source,
            synthetic_only=data.source == "synthetic_only",
            folds_completed=config.fold_ids,
        )
        write_csv(run_dir / "metrics_by_window.csv", all_rows)
        write_csv(run_dir / "metrics_by_participant.csv", participants)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "status.json", {"status": "complete", "folds_completed": config.fold_ids})
    except BaseException as error:
        write_json(
            run_dir / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "current_fold": fold,
                "folds_completed": completed,
            },
        )
        raise
    finally:
        tf.keras.backend.clear_session()
    print(f"Saved run: {run_dir}", flush=True)
    return run_dir


def run_demo(config):
    config = demo_configuration(config)
    return run_experiment(config, synthetic_dataset(config))
