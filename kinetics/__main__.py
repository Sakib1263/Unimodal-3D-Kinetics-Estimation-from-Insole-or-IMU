"""One entry point for inspection, training, a data-free demo, and plots.

Examples: python -m kinetics demo; python -m kinetics train --config configs/km2kn.json
Use ``python -m kinetics COMMAND --help`` for command-specific arguments.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

from .config import ExperimentConfig
from .runner import inspect_dataset, run_experiment, run_demo


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Unimodal plantar kinetics research workflows.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("inspect", "Validate real inputs and print planned participant splits."),
        ("train", "Train and evaluate the configured folds."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--config", required=True, help="Path to an experiment JSON file.")
        if name == "train":
            command.add_argument("--epochs", type=int, help="Override maximum training epochs.")
    demo = commands.add_parser("demo", help="Run one small synthetic fold without research data.")
    demo.add_argument("--config", help="Optional base configuration; synthetic dimensions override it.")
    demo.add_argument("--output-dir", help="Optional run directory (relative to project_dir or absolute).")
    commands.add_parser("plot", help="Plot a saved reference/prediction NPZ archive.", add_help=False)
    commands.add_parser(
        "raw-plot", help="Plot participant-balanced stances from source CSVs.", add_help=False
    )
    try:
        # Plot parsers live with plotting, so notebooks and CLI use the same schema.
        if argv and argv[0] in ("plot", "raw-plot"):
            from .plotting import plot_cli, raw_plot_cli

            return (plot_cli if argv[0] == "plot" else raw_plot_cli)(argv[1:])
        args = parser.parse_args(argv)
        config = (
            ExperimentConfig.from_json(args.config)
            if args.config
            else ExperimentConfig(project_dir=str(Path.cwd()))
        )
        if args.command == "inspect":
            print(json.dumps(inspect_dataset(config), indent=2, allow_nan=False))
        elif args.command == "demo":
            if args.output_dir:
                config = replace(config, output_dir=args.output_dir)
            run_demo(config)
        else:
            if args.epochs is not None:
                config = replace(config, epochs=args.epochs)
            run_experiment(config)
    except (ValueError, OSError, BadZipFile, EOFError) as error:
        parser.exit(2, f"Configuration/data error: {error}\n")


if __name__ == "__main__":
    main()
