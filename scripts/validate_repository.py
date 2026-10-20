"""Check source syntax, notebook hygiene, model dispatch, configs and local links.

Run from any directory: python <checkout>/scripts/validate_repository.py
This static check neither loads study data nor imports TensorFlow.
"""

import ast
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kinetics.config import ExperimentConfig

NOTEBOOKS = {"Km2Kn.ipynb", "Insole2Kn3D_GRFM.ipynb", "Insole2Kn3D_CoP.ipynb"}
BAD_ENCODING = re.compile(r"\ufffd|[\u0080-\u009f]|[\u00c2\u00c3][\u00a0-\u00bf]|\u00e2[\u2000-\u21ff]")


def check_text_encoding(text, label):
    """Catch replacement characters and common UTF-8/Windows-1252 mix-ups."""
    if text.startswith("\ufeff") or BAD_ENCODING.search(text):
        raise ValueError(f"Invalid or corrupted text encoding in {label}; save as UTF-8 without a BOM.")


def check_file_encoding(path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Expected UTF-8 text in {path.relative_to(ROOT)}") from error
    check_text_encoding(text, path.relative_to(ROOT))


def check_links(text, parent, label):
    """Resolve repository-local Markdown links; external links are not fetched."""
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.strip().strip("<>").split("#", 1)[0]
        if not target or re.match(r"[a-zA-Z]+:", target):
            continue
        if not (parent / unquote(target)).exists():
            raise ValueError(f"Broken local link in {label}: {target}")


def main():
    python_files = sorted(
        [
            *ROOT.joinpath("kinetics").glob("*.py"),
            *ROOT.joinpath("models").glob("*.py"),
            *ROOT.joinpath("scripts").glob("*.py"),
        ]
    )
    trees = {p: ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p)) for p in python_files}
    actual = {p.name for p in ROOT.glob("*.ipynb")}
    if actual != NOTEBOOKS:
        raise ValueError(f"Expected only the three maintained notebooks; found {sorted(actual)}")
    for filename in sorted(NOTEBOOKS):
        check_file_encoding(ROOT / filename)
        notebook = nbformat.read(ROOT / filename, as_version=4)
        nbformat.validate(notebook)
        headings = []
        for index, cell in enumerate(notebook.cells):
            label = f"{filename}:cell{index}"
            check_text_encoding(cell.source, label)
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=label)
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    raise ValueError(f"Clear execution outputs before sharing: {label}")
            else:
                check_links(cell.source, ROOT, label)
                headings += re.findall(r"^## (\d+)\.", cell.source, re.MULTILINE)
        if headings != [str(i) for i in range(1, 11)]:
            raise ValueError(f"Notebook sections must follow steps 1-10: {filename}")
    configs = list(ROOT.joinpath("configs").glob("*.json"))
    for path in configs:
        ExperimentConfig.from_json(path)
    registry = next(
        n.value
        for n in trees[ROOT / "kinetics/modeling.py"].body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MODEL_REGISTRY" for t in n.targets)
    )
    for name, (module, class_name, method) in ast.literal_eval(registry).items():
        ExperimentConfig(model=name)
        tree = trees[ROOT / "models" / (module + ".py")]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        if not any(isinstance(n, ast.FunctionDef) and n.name == method for n in cls.body):
            raise ValueError(f"Missing registered architecture method: {name}")
    markdown = [
        *ROOT.glob("*.md"),
        *ROOT.joinpath("docs").glob("*.md"),
        *ROOT.joinpath("models").glob("*.md"),
        *ROOT.joinpath("configs").glob("*.md"),
        *ROOT.joinpath("outputs").glob("*.md"),
    ]
    for path in markdown:
        check_links(path.read_text(encoding="utf-8-sig"), path.parent, str(path.relative_to(ROOT)))
    text_files = {
        *python_files,
        *markdown,
        *configs,
        *ROOT.glob("*.txt"),
        *ROOT.joinpath(".github").rglob("*.yml"),
        *ROOT.joinpath("outputs").rglob("*.csv"),
        *(
            ROOT / name
            for name in (
                "Data_Preprocessing.m",
                "pyproject.toml",
                "LICENSE",
                ".gitattributes",
                ".gitignore",
                ".editorconfig",
            )
        ),
    }
    for path in text_files:
        check_file_encoding(path)
    print(
        f"Validated {len(python_files)} Python files, {len(NOTEBOOKS)} clean notebooks, "
        f"{len(configs)} configurations, model dispatch, {len(markdown)} Markdown files and UTF-8 text encoding."
    )


if __name__ == "__main__":
    main()
