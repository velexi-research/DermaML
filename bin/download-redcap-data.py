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
import pandas as pd
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

# ------ Error codes

USAGE_ERROR_EXIT_CODE = 2
RUNTIME_ERROR_EXIT_CODE = 3

# ------ REST API parameters

REDCAP_API_URL = "https://redcap.icts.uiowa.edu/redcap/api/"
DEFAULT_REQUESTS_DELAY = 0.2  # seconds. The REDCap request limit is 600 requests / min

# ------ Dataset parameters

REDCAP_RECORD_FIELD_NAMES = [
    "record_id",
    "gender",
    "birth_year",
    "sex_assigned_at_birth",
    "race_ethnicity",
    "please_specify",
    "ethnicity",
    "handedness",
    "occupation",
    "driving_time",
    "sun_exposure",
    "sunscreen_use",
    "state",
    "please_specify4",
    "photo",
    "right_hand_dorsal_surface",
    "form_1_complete",
]

REDCAP_RECORD_DTYPES = {
    "record_id": int,
    "gender": int,
    "birth_year": int,
    "sex_assigned_at_birth": int,
    "race_ethnicity___1": int,
    "race_ethnicity___2": int,
    "race_ethnicity___3": int,
    "race_ethnicity___4": int,
    "race_ethnicity___5": int,
    "race_ethnicity___6": int,
    "ethnicity": int,
    "handedness": int,
    "occupation": int,
    "driving_time": int,
    "sun_exposure": int,
    "sunscreen_use": int,
    "state": int,
    "form_1_complete": int,
}


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
    "form_1_complete": "form_complete",
}

REDCAP_LEFT_HAND_IMAGE_FIELD_NAME = "photo"
REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME = "right_hand_dorsal_surface"

# field value mappings
FORM_COMPLETED_VALUE = "2"

GENDER_MAP = {
    1: "Man",
    2: "Woman",
    3: "Gender queer or gender fluid",
    4: "Non-binary or not exclusively man or woman",
}

SEX_ASSIGNED_AT_BIRTH_MAP = {
    1: "Male",
    2: "Female",
    5: "Prefer not to say",
}

ETHNICITY_MAP = {
    1: "Hispanic/Latino",
    2: "Not Hispanic/Latino",
}

HANDEDNESS_MAP = {
    1: "Right",
    2: "Left",
    3: "Both",
}

OCCUPATION_MAP = {
    1: "Architecture/Engineering",
    2: "Building/Ground Cleaning and Maintenance",
    3: "Community and Social Services",
    4: "Computer/Mathematical",
    5: "Construction",
    6: "Education, Training, Library",
    7: "Farming/Fishing/Forestry",
    8: "Food Preparation/Serving",
    9: "Healthcare/Medical",
    10: "Installation/Maintenance/Repair",
    11: "Legal",
    12: "Office and Administrative Support",
    13: "TBD",
    14: "Sales/Marketing",
    15: "Transportation",
    16: "Caretaking/Stay at Home Parent",
    17: "Student",
    18: "Unemployed",
}

DRIVING_TIME_MAP = {
    1: "30 minutes or less",
    2: "30-60 minutes",
    3: "60+ minutes",
}

SUN_EXPOSURE_MAP = {
    1: "less than 30 minutes",
    2: "1 hour",
    3: "TBD",
    4: "TBD",
    5: "TBD",
    6: "2 hours",
    7: "3 hours",
    8: "4 hours",
    9: "5+ hours",
}

SUNSCREEN_USE_MAP = {
    1: "Almost always",
    2: "Sometimes",
    3: "Rarely",
    4: "Never",
}

STATE_MAP = {
    1: "Pacific (WA, OR, CA)",
    2: "Rocky Mountains (ID, MT, WY, NV, UT, CO)",
    3: "Southwest (AZ, NM, TX, OK)",
    4: "Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, IN, MI, OH)",
    5: "Southeast (AR, LA, MS, TN, KY, WV, VA, NC, SC, GA, AL, FL, DL, MD, DC)",
    6: "Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME)",
    7: "Noncontiguous (HI, AK)",
    8: "Other",
}

FORM_COMPLETE_MAP = {
    0: "Incomplete",
    2: "Completed",
}

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

    # Configure Pandas
    pd.options.mode.copy_on_write = True

    # Load configuration
    config = load_config(config_file)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    image_dir = output_dir / "images"
    os.makedirs(image_dir, exist_ok=True)

    # --- Retrieve raw metadata

    # Emit info message
    typer.echo("Retrieving metadata...")

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

    # Emit info message
    typer.echo(f"Successfully retrieved metadata in {time.time() - t_start:0.2f}s\n")

    # --- Clean metadata

    # Emit info message
    typer.echo("Cleaning metadata...")

    # Start timer
    t_start = time.time()
    # Load response data into a DataFrame
    records = json.loads(response.text)
    raw_data = DataFrame(records)

    # Initialize cleaned data and invalid records DataFrames
    cleaned_data = raw_data.copy()
    invalid_records = DataFrame(columns=[*raw_data.columns, "error"])

    # Remove incomplete records
    incomplete_records = cleaned_data[
        cleaned_data["form_1_complete"] != FORM_COMPLETED_VALUE
    ]
    incomplete_records["error"] = "incomplete record"
    invalid_records = pd.concat([invalid_records, incomplete_records]).reset_index(
        drop=True
    )
    cleaned_data.drop(index=incomplete_records.index, inplace=True)

    # Remove records with invalid birth year
    invalid_birth_year = cleaned_data[
        cleaned_data["birth_year"].str.isnumeric() == False
    ]
    invalid_birth_year["error"] = "invalid birth year"
    invalid_records = pd.concat([invalid_records, invalid_birth_year]).reset_index(
        drop=True
    )
    cleaned_data.drop(index=invalid_birth_year.index, inplace=True)

    # Fix column dtypes
    for column, dtype in REDCAP_RECORD_DTYPES.items():
        cleaned_data[column] = cleaned_data[column].astype(dtype)

    # Construct metadata DataFrame
    metadata = DataFrame()

    metadata["record_id"] = cleaned_data["record_id"]

    metadata["gender"] = cleaned_data["gender"].map(lambda x: GENDER_MAP[int(x)])

    metadata["birth_year"] = cleaned_data["birth_year"]

    metadata["sex_assigned_at_birth"] = cleaned_data["sex_assigned_at_birth"].map(
        lambda x: SEX_ASSIGNED_AT_BIRTH_MAP[int(x)]
    )

    metadata["race_ethnicity"] = cleaned_data["race_ethnicity___1"]  # TODO

    metadata["ethnicity"] = cleaned_data["ethnicity"].map(
        lambda x: ETHNICITY_MAP[int(x)]
    )

    metadata["handedness"] = cleaned_data["handedness"].map(
        lambda x: HANDEDNESS_MAP[int(x)]
    )

    metadata["occupation"] = cleaned_data["occupation"].map(
        lambda x: OCCUPATION_MAP[int(x)]
    )

    metadata["driving_time"] = cleaned_data["driving_time"].map(
        lambda x: DRIVING_TIME_MAP[int(x)]
    )

    metadata["sun_exposure"] = cleaned_data["sun_exposure"].map(
        lambda x: SUN_EXPOSURE_MAP[int(x)]
    )

    metadata["sunscreen_use"] = cleaned_data["sunscreen_use"].map(
        lambda x: SUNSCREEN_USE_MAP[int(x)]
    )

    metadata["state"] = cleaned_data["state"].map(lambda x: STATE_MAP[int(x)])

    metadata["left_hand_image_file"] = cleaned_data[REDCAP_LEFT_HAND_IMAGE_FIELD_NAME]
    metadata["right_hand_image_file"] = cleaned_data[REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME]

    metadata["form_complete"] = cleaned_data["form_1_complete"].map(
        lambda x: FORM_COMPLETE_MAP[int(x)]
    )

    # Save metadata to file
    raw_data.to_csv(output_dir / "raw_data.csv", index=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    invalid_records.to_csv(output_dir / "invalid_records.csv", index=False)

    # Emit info message
    typer.echo(
        f"{len(raw_data)} records processed. "
        f"(valid: {len(metadata)}, invalid: {len(invalid_records)})"
    )
    typer.echo(f"Finished cleaning metadata in {time.time() - t_start:0.2f}s\n")

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
            "field": REDCAP_LEFT_HAND_IMAGE_FIELD_NAME,
        }
        response = post_request_with_delay(REDCAP_API_URL, data)

        # Check status code
        if not response.ok:
            _raise_runtime_error(
                f"Error retrieving from left hand image for record '{record[record_id]}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Save image
        image_path = image_dir / record["right_hand_image_file"]
        if not os.path.isfile(image_path):
            with open(image_path, "wb") as file:
                file.write(response.content)
                file.close()
        else:
            # TODO: raise exception
            pass

        # --- Retrieve right hand image

        # Send request
        data = {
            "token": config["api_token"],
            "content": "file",
            "action": "export",
            "record": record["record_id"],
            "field": REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME,
        }
        response = post_request_with_delay(REDCAP_API_URL, data)

        # Check status code
        if not response.ok:
            _raise_runtime_error(
                f"Error retrieving from right hand image for record '{record[record_id]}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Save image
        image_path = image_dir / record["right_hand_image_file"]
        if not os.path.isfile(image_path):
            with open(image_path, "wb") as file:
                file.write(response.content)
                file.close()
        else:
            # TODO: raise exception
            pass

    # Emit info message
    typer.echo(f"Successfully retrieved images in {time.time() - t_start:0.2f}s\n")


# --- Run app

if __name__ == "__main__":
    typer.run(main)
