#!/usr/bin/env python
#
#   Copyright 2024 Velexi Corporation
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Script for filtering out records with non-standard image files.
"""
# --- Imports


# Standard library
from pathlib import Path
import os
import time
from typing import Annotated

# External packages
import pandas as pd
from pandas import DataFrame
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from typer.rich_utils import (
    highlighter,
    STYLE_ERRORS_PANEL_BORDER,
    ERRORS_PANEL_TITLE,
    ALIGN_ERRORS_PANEL,
)
import skimage
import typer


# --- Constants

# ------ Error codes

USAGE_ERROR_EXIT_CODE = 2
RUNTIME_ERROR_EXIT_CODE = 3

# --- Helper functions


def _raise_cli_argument_error(message: str) -> None:
    """
    Display CLI argument/option error and exit with "usage error" code (defined by `Click`
    package).

    Parameters
    ----------
    `message`: error message
    """
    console = Console(stderr=True)
    console.print(
        Panel(
            highlighter(message),
            border_style=STYLE_ERRORS_PANEL_BORDER,
            title=ERRORS_PANEL_TITLE,
            title_align=ALIGN_ERRORS_PANEL,
        )
    )
    raise typer.Exit(code=USAGE_ERROR_EXIT_CODE)


def _raise_runtime_error(message: str) -> None:
    """
    Display runtime error and exit with an error code.

    Parameters
    ----------
    `message`: error message
    """
    console = Console(stderr=True)
    console.print(
        Panel(
            highlighter(message),
            border_style=STYLE_ERRORS_PANEL_BORDER,
            title=ERRORS_PANEL_TITLE,
            title_align=ALIGN_ERRORS_PANEL,
        )
    )
    raise typer.Exit(code=RUNTIME_ERROR_EXIT_CODE)


# ------ argument/option validators


def _validate_file_exists(path: str) -> str:
    """
    Validate that `path` exists and is a file.

    Parameters
    ----------
    `path`: file path to validate

    Return Value
    ------------
    validated path
    """
    if not os.path.exists(path):
        message = f"File '{path}' does not exist."
        _raise_cli_argument_error(message)

    if not os.path.isfile(path):
        message = f"File '{path}' is not a file."
        _raise_cli_argument_error(message)

    return path


def _validate_path_does_not_exist(path: str) -> str:
    """
    Validate that `path` does not exist.

    Parameters
    ----------
    `path`: path to validate

    Return Value
    ------------
    validated path
    """
    if os.path.isdir(path):
        message = f"'{path}' is a directory that already exists."
        _raise_cli_argument_error(message)

    if os.path.isfile(path):
        message = f"'{path}' is a file that already exists."
        _raise_cli_argument_error(message)

    return path


# ------ data processing functions

# --- CLI arguments and options


METADATA_INFILE_ARG = typer.Argument(
    ...,
    help=(
        "Original metadata file. Image are assumed to be in a directory named 'images'"
        "in the same directory that the metadata file resides in."
    ),
    callback=_validate_file_exists,
)

DEFAULT_METADATA_OUTFILE = "metadata-cleaned.csv"
METADATA_OUTFILE_OPTION = typer.Option(
    "-o",
    help="File to save cleaned metadata to.",
    callback=_validate_path_does_not_exist,
)


DEFAULT_ERRORS_OUTFILE = "errors.csv"
ERRORS_OUTFILE_OPTION = typer.Option(
    "-e",
    help="File to save errors to. If the error file already exists, it is appended to.",
)


# --- Helper functions


# --- Main program


def main(
    metadata_infile: Annotated[Path, METADATA_INFILE_ARG],
    metadata_outfile: Annotated[
        Path, METADATA_OUTFILE_OPTION
    ] = DEFAULT_METADATA_OUTFILE,
    errors_outfile: Annotated[Path, ERRORS_OUTFILE_OPTION] = DEFAULT_ERRORS_OUTFILE,
) -> None:
    """
    Filter out records with non-standard image files.
    """
    # --- Preparations

    # Configure Pandas
    pd.options.mode.copy_on_write = True

    images_dir = metadata_infile.parent / "images"
    if not images_dir.is_dir():
        _raise_runtime_error(f"Cannot find images directory '{ images_dir }'.")

    # --- Load metadata

    # Emit info message
    typer.echo("Load metadata...")

    metadata = pd.read_csv(metadata_infile)

    # Emit info message
    typer.echo("Successfully loaded metadata\n")

    # --- Clean metadata

    # Emit info message
    typer.echo("Cleaning metadata...")

    # Start timer
    t_start = time.time()

    # Initialize record arrays
    cleaned_metadata_records = []
    error_records = []

    for idx in track(
        range(len(metadata)), description="Checking image file formats..."
    ):

        # --- Preparations

        # Get record
        record = metadata.iloc[idx, :]

        # --- Check if image files for the record are in the standard format

        # Check left hand image
        image_path = images_dir / record["left_hand_image_file"]
        left_hand_check_failed = False
        try:
            skimage.io.imread(image_path)
        except OSError:
            left_hand_check_failed = True

        # Check right hand image
        image_path = images_dir / record["right_hand_image_file"]
        right_hand_check_failed = False
        try:
            skimage.io.imread(image_path)
        except OSError:
            right_hand_check_failed = True

        # Process error
        if left_hand_check_failed or right_hand_check_failed:
            if left_hand_check_failed and right_hand_check_failed:
                record["reason"] = "unsupported image file format for both hands"
            elif left_hand_check_failed:
                record["reason"] = "unsupported image file format for left hand"
            else:  # right_hand_check_failed
                record["reason"] = "unsupported image file format for right hand"

            error_records.append(record)
            continue

        # Both hand images pass, so append record to cleaned metadata list
        cleaned_metadata_records.append(record)

    # Construct DataFrame containing cleaned metadata and errors
    cleaned_metadata = DataFrame(cleaned_metadata_records)
    errors = DataFrame(error_records)

    # Emit info message
    typer.echo(
        f"{len(metadata)} records processed. "
        f"(valid: {len(cleaned_metadata)}, errors: {len(errors)})"
    )
    typer.echo(f"Finished cleaning metadata in {time.time() - t_start:0.2f}s\n")

    # --- Save cleaned metadata and errors to CSV files

    cleaned_metadata.to_csv(metadata_outfile, index=False)

    if not errors_outfile.is_file():
        errors.to_csv(errors_outfile, index=False, mode="w")
    else:
        errors.to_csv(errors_outfile, index=False, header=False, mode="a")


# --- Run app

if __name__ == "__main__":
    typer.run(main)
