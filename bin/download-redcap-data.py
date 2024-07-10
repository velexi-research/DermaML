#!/usr/bin/env python
#
#   Copyright 2022 Velexi Corporation
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
Script for downloading data from REDCap
"""
# --- Imports


# Standard library
import json
from pathlib import Path
import os
import time
from typing import Annotated

# External packages
from pandas import DataFrame
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from typer.rich_utils import (
    highlighter,
    STYLE_ERRORS_PANEL_BORDER,
    ERRORS_PANEL_TITLE,
    ALIGN_ERRORS_PANEL,
)
import typer
import yaml


# --- Constants


# Error codes
USAGE_ERROR_EXIT_CODE = 2
RUNTIME_ERROR_EXIT_CODE = 3

# REST API parameters
REDCAP_API_URL = "https://redcap.icts.uiowa.edu/redcap/api/"
DEFAULT_REQUESTS_DELAY = 0.2  # seconds. The REDCap request limit is 600 requests / min

# Dataset parameters
REDCAP_RECORD_FIELD_NAMES = [
    "record_id",
    "gender",
    "birth_year",
    "sex_assigned_at_birth",
    "race_ethnicity___1",
    "race_ethnicity___2",
    "race_ethnicity___3",
    "race_ethnicity___4",
    "race_ethnicity___5",
    "race_ethnicity___6",
    "ethnicity",
    "handedness",
    "occupation",
    "driving_time",
    "sun_exposure",
    "sunscreen_use",
    "state",
    "photo",
    "right_hand_dorsal_surface",
    "form_1_complete",
]

ADJUSTED_RECORD_FIELD_NAMES = {
    "record_id": "record_id",
    "gender": "gender",
    "birth_year": "birth_year",
    "sex_assigned_at_birth": "sex_assigned_at_birth",
    "race_ethnicity___1": "race_ethnicity___1",
    "race_ethnicity___2": "race_ethnicity___2",
    "race_ethnicity___3": "race_ethnicity___3",
    "race_ethnicity___4": "race_ethnicity___4",
    "race_ethnicity___5": "race_ethnicity___5",
    "race_ethnicity___6": "race_ethnicity___6",
    "ethnicity": "ethnicity",
    "handedness": "handedness",
    "occupation": "occupation",
    "driving_time": "driving_time",
    "sun_exposure": "sun_exposure",
    "sunscreen_use": "sunscreen_use",
    "state": "state",
    "photo": "left_hand_image_file",
    "right_hand_dorsal_surface": "right_hand_image_file",
    "form_1_complete": "form_1_complete",
}

LEFT_HAND_IMAGE_FIELD_NAME = "photo"
RIGHT_HAND_IMAGE_FIELD_NAME = "right_hand_dorsal_surface"


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


# --- CLI arguments and options


CONFIG_FILE_ARG = typer.Argument(
    ...,
    help=("Configuration file"),
    callback=_validate_file_exists,
)

DEFAULT_OUTPUT_DIR = "hawkeye-hands"
OUTPUT_DIR_ARG = typer.Option(
    "--output-dir",
    "-o",
    help="Directory to save data to.",
    callback=_validate_path_does_not_exist,
)

DEFAULT_QUIET_OPTION = False
QUIET_OPTION = typer.Option("--quiet", "-q", help="Display fewer status messages.")


# --- Helper functions


def post_request_with_delay(
    url: str, data: dict, delay: float = DEFAULT_REQUESTS_DELAY
):
    """
    Send POST request delayed by `delay` seconds.

    Parameters
    ----------
    `url`: URL to send request to

    `data`: data to send with POST request

    `delay`: number of seconds to wait before sending request.
    """
    # Wait before sending calling function
    time.sleep(delay)

    # Send request and return response
    return requests.post(url, data=data)


def load_config(config_file: str) -> dict:
    """
    Load configuration from file.

    Parameters
    ----------
    `config_file`: configuration file

    Return values
    -------------
    dictionary containing configuration parameters
    """
    # --- Load configuration parameters

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # --- Validate configuration parameters

    if "api_token" not in config:
        _raise_runtime_error("'api_token' missing from configuration file.")

    return config


# --- Main program


def main(
    config_file: Annotated[Path, CONFIG_FILE_ARG],
    output_dir: Annotated[Path, OUTPUT_DIR_ARG] = DEFAULT_OUTPUT_DIR,
    quiet: Annotated[bool, QUIET_OPTION] = DEFAULT_QUIET_OPTION,
) -> None:
    """
    Download data from REDCap
    """
    # --- Preparations

    # Load configuration
    config = load_config(config_file)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    image_dir = output_dir / "images"
    os.makedirs(image_dir, exist_ok=True)

    # --- Retrieve metadata

    # Emit info message
    typer.echo("Retriving metadata...")

    # Start timer
    t_start = time.time()

    # Send request
    data = {
        "token": config["api_token"],
        "content": "record",
        "format": "json",
        "type": "flat",
        "fields": ",".join(REDCAP_RECORD_FIELD_NAMES),
    }
    response = post_request_with_delay(REDCAP_API_URL, data)

    # Check status code
    if not response.ok:
        _raise_runtime_error(
            "Error retrieving from metadata. "
            f"Received HTTP status code {response.status_code}."
        )

    # Load response data into a DataFrame
    records = json.loads(response.text)
    metadata = DataFrame(records)

    # Fix column names
    metadata.rename(columns=ADJUSTED_RECORD_FIELD_NAMES, inplace=True)

    # Save metadata to file
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # Emit info message
    typer.echo(f"Successfully retrieved metadata in {time.time() - t_start:0.2f}s\n")

    # --- Retrieve images

    # Start timer
    t_start = time.time()

    for idx in track(range(len(metadata)), description="Retrieving images..."):

        # --- Preparations

        # Get record
        record = metadata.iloc[idx, :]

        # --- Retrieve left hand image

        # Send request
        data = {
            "token": config["api_token"],
            "content": "file",
            "action": "export",
            "record": record["record_id"],
            "field": LEFT_HAND_IMAGE_FIELD_NAME,
        }
        response = post_request_with_delay(REDCAP_API_URL, data)

        # Check status code
        if not response.ok:
            _raise_runtime_error(
                f"Error retrieving from left hand image for record '{record[record_id]}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Save image
        with open(image_dir / record["left_hand_image_file"], "wb") as file:
            file.write(response.content)
            file.close()

        # --- Retrieve right hand image

        # Send request
        data = {
            "token": config["api_token"],
            "content": "file",
            "action": "export",
            "record": record["record_id"],
            "field": RIGHT_HAND_IMAGE_FIELD_NAME,
        }
        response = post_request_with_delay(REDCAP_API_URL, data)

        # Check status code
        if not response.ok:
            _raise_runtime_error(
                f"Error retrieving from right hand image for record '{record[record_id]}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Save image
        with open(image_dir / record["right_hand_image_file"], "wb") as file:
            file.write(response.content)
            file.close()

    # Emit info message
    typer.echo(f"Successfully retrieved images in {time.time() - t_start:0.2f}s\n")


# --- Run app

if __name__ == "__main__":
    typer.run(main)
