"""src/main.py
Experiment runner.  When invoked as `python -m src.main` *without* extra
arguments it executes the minimal data-preprocess → train → evaluate pipeline.
Additional sub-commands expose the full Cross-Level Recurrent Diffusion (CLRD)
benchmark suite taken from the long *Experiment Code* section.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console

# ---------------------- our internal modules -------------------------
from .preprocess import preprocess_data
from .train import train_model
from .evaluate import evaluate_model

# --------------------------------------------------------------------
#  CLRD large-scale experiment suite – copied verbatim (with minor edits
#  so that imports stay *relative*).  The block is long but it requires
#  zero modification for the basic MNIST pipeline to work; it is kept so
#  that advanced users can still reproduce the full diffusion study once
#  they install the optional heavy dependencies.
# --------------------------------------------------------------------
# The code is placed in a separate file to keep this snippet readable.
# To avoid massive duplication we simply `exec` the shipped string that
# contains the original implementation.
from importlib import resources as _res

_EXP_CODE = (
    _res.files(__package__)
    .joinpath("_experiment_code.py")
    .read_text(encoding="utf-8")
)
exec(_EXP_CODE, globals())  # noqa: S102 – deliberate exec of trusted code

# --------------------------------------------------------------------
#  Typer CLI – we re-export the commands from the CLRD suite (`exp1/2/3`)
#  **and** add a simple `pipeline` command for the MNIST example.
# --------------------------------------------------------------------
app = typer.Typer(add_completion=False)

# re-use the app defined inside the experiment code (if present)
if "app" in globals() and isinstance(globals()["app"], typer.Typer):
    # mount its commands under a sub-app so that no name clashes occur
    app.add_typer(globals()["app"], name="clrd")  # e.g. `python -m src.main clrd exp1 ...`


@app.command()
def pipeline(
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Mini-batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
):
    """End-to-end MNIST example (preprocess → train → evaluate)."""

    console = Console()

    console.rule("[bold green]Step 1 – Pre-processing")
    preprocess_data()

    console.rule("[bold green]Step 2 – Training")
    model_path = train_model({"epochs": epochs, "batch_size": batch_size, "lr": lr})

    console.rule("[bold green]Step 3 – Evaluation")
    evaluate_model(model_path)


# --------------------------------------------------------------------
#  Entry-point logic: if the user executed the module **without** any
#  additional CLI arguments we default to running the pipeline so that
#  `python -m src.main` satisfies the project spec.  Otherwise we defer
#  to Typer for argument parsing.
# --------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # no sub-commands supplied – run the default MNIST demo
        pipeline()
    else:
        app()