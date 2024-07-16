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
import uuid

# External packages
import numpy as np
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
    "please_specificy",
    "birth_year",
    "sex_assigned_at_birth",
    "please_specify2",
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

REDCAP_LEFT_HAND_IMAGE_FIELD_NAME = "photo"
REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME = "right_hand_dorsal_surface"

# field value mappings
FORM_COMPLETED_VALUE = "2"

GENDER_MAP = {
    1: "Man",
    2: "Woman",
    3: "Gender queer or gender fluid",
    4: "Non-binary or not exclusively man or woman",
    5: "Questioning or exploring",
    6: "None of these describe me",
    7: "Prefer not to answer",
}

SEX_ASSIGNED_AT_BIRTH_MAP = {
    1: "Male",
    2: "Female",
    3: "Intersex or variation of sex characteristics",
    4: "None of these describe me",
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
    13: "Protective Services",
    14: "Sales/Marketing",
    15: "Transportation",
    16: "Caretaking/Stay at Home Parent",
    17: "Student",
    18: "Unemployed",
}

AVERAGE_DRIVING_TIME_MAP = {
    1: "30 minutes or less",
    2: "30-60 minutes",
    3: "60+ minutes",
}

SUN_EXPOSURE_MAP = {
    1: "less than 30 minutes",
    2: "1 hour",
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
    1: "Unverified",
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


# ------ data processing functions


def _process_race_ethnicity(record) -> str:
    """
    Construct comma-separated list of race/ethnicity for record.

    Parameters
    ----------
    `record`: row of DataFrame

    Return Value
    ------------
    comma-separated list of races/ethnicities for individual
    """
    race_ethnicity = ""

    if record["race_ethnicity___1"] == 1:
        race_ethnicity = ",".join([race_ethnicity, "Caucasian/White"])

    if record["race_ethnicity___2"] == 1:
        race_ethnicity = ",".join([race_ethnicity, "African-American/Black"])

    if record["race_ethnicity___3"] == 1:
        race_ethnicity = ",".join([race_ethnicity, "Asian"])

    if record["race_ethnicity___4"] == 1:
        race_ethnicity = ",".join([race_ethnicity, "Native Hawaiian/Pacific Islander"])

    if record["race_ethnicity___5"] == 1:
        race_ethnicity = ",".join([race_ethnicity, "American Indian/Alaska National"])

    return race_ethnicity.strip(",")


def _clean_metadata(raw_data_: DataFrame) -> (DataFrame, DataFrame):
    """
    Clean metadata.

    Parameters
    ----------
    `raw_data`: raw data

    Return Value
    ------------
    `clean_data`: DataFrame containing cleaned records

    `invalid_data`: DataFrame containing invalid records
    """
    # Initialize valid records and invalid records DataFrames
    valid_records = raw_data_.copy()
    invalid_data = DataFrame(columns=[*raw_data_.columns, "reason"])

    # Remove incomplete records
    incomplete_records = valid_records[
        valid_records["form_1_complete"] != FORM_COMPLETED_VALUE
    ]
    incomplete_records["reason"] = "incomplete record"
    invalid_data = pd.concat([invalid_data, incomplete_records]).reset_index(drop=True)
    valid_records.drop(index=incomplete_records.index, inplace=True)

    # Remove records with invalid birth year
    invalid_birth_year = valid_records[
        valid_records["birth_year"].str.isnumeric() == False  # noqa
    ]
    invalid_birth_year["reason"] = "invalid birth year"
    invalid_data = pd.concat([invalid_data, invalid_birth_year]).reset_index(drop=True)
    valid_records.drop(index=invalid_birth_year.index, inplace=True)

    # Remove records with missing image data
    invalid_images = valid_records[
        np.logical_or(
            valid_records[REDCAP_LEFT_HAND_IMAGE_FIELD_NAME].map(lambda x: x.strip())
            == "",  # noqa
            valid_records[REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME].map(lambda x: x.strip())
            == "",  # noqa
        )
    ]
    invalid_images["reason"] = "missing images"
    invalid_data = pd.concat([invalid_data, invalid_images]).reset_index(drop=True)
    valid_records.drop(index=invalid_images.index, inplace=True)

    # Remove dashes from race/ethnicity "Please Specify" column
    valid_records["please_specify"] = valid_records["please_specify"].map(
        lambda x: "" if x == "-" else x
    )

    # Fix column dtypes
    for column, dtype in REDCAP_RECORD_DTYPES.items():
        valid_records[column] = valid_records[column].astype(dtype)

    # Construct clean_data DataFrame
    clean_data = DataFrame()

    clean_data["record_id"] = valid_records["record_id"]

    clean_data["gender"] = valid_records["gender"].map(lambda x: GENDER_MAP[int(x)])

    clean_data["gender_specify"] = valid_records["please_specificy"].map(
        lambda x: x.strip()
    )

    clean_data["birth_year"] = valid_records["birth_year"]

    clean_data["sex_assigned_at_birth"] = valid_records["sex_assigned_at_birth"].map(
        lambda x: SEX_ASSIGNED_AT_BIRTH_MAP[int(x)]
    )

    clean_data["sex_assigned_at_birth_specify"] = valid_records["please_specify2"].map(
        lambda x: x.strip()
    )

    clean_data["race_ethnicity"] = valid_records.apply(_process_race_ethnicity, axis=1)

    clean_data["race_ethnicity_specify"] = valid_records["please_specify"].map(
        lambda x: x.strip()
    )

    clean_data["ethnicity"] = valid_records["ethnicity"].map(
        lambda x: ETHNICITY_MAP[int(x)]
    )

    clean_data["handedness"] = valid_records["handedness"].map(
        lambda x: HANDEDNESS_MAP[int(x)]
    )

    clean_data["occupation"] = valid_records["occupation"].map(
        lambda x: OCCUPATION_MAP[int(x)]
    )

    clean_data["driving_time"] = valid_records["driving_time"].map(
        lambda x: AVERAGE_DRIVING_TIME_MAP[int(x)]
    )

    clean_data["sun_exposure"] = valid_records["sun_exposure"].map(
        lambda x: SUN_EXPOSURE_MAP[int(x)]
    )

    clean_data["sunscreen_use"] = valid_records["sunscreen_use"].map(
        lambda x: SUNSCREEN_USE_MAP[int(x)]
    )

    clean_data["region"] = valid_records["state"].map(lambda x: STATE_MAP[int(x)])

    clean_data["region_specify"] = valid_records["please_specify4"].map(
        lambda x: x.strip()
    )

    clean_data["left_hand_image_file"] = valid_records[
        REDCAP_LEFT_HAND_IMAGE_FIELD_NAME
    ]
    clean_data["right_hand_image_file"] = valid_records[
        REDCAP_RIGHT_HAND_IMAGE_FIELD_NAME
    ]

    clean_data["form_complete"] = valid_records["form_1_complete"].map(
        lambda x: FORM_COMPLETE_MAP[int(x)]
    )

    return clean_data, invalid_data


def _download_images(config: dict, image_dir: Path, metadata: DataFrame) -> None:
    """
    Download image data.

    Parameters
    ----------
    `config`: dictionary containing configuration parameters

    `image_dir`: path to directory where images should be saved

    `metadata`: DataFrame containing records to retrieve images for

    Return Value
    ------------
    None
    """
    # Initialize list of image UUIDs
    image_uuids = []

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
            record_id = record["record_id"]
            _raise_runtime_error(
                f"Error retrieving from left hand image for record '{record_id}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Check if file name is a valid UUID
        image_uuid_str = os.path.splitext(record["left_hand_image_file"])[0]
        image_uuid_invalid = False
        try:
            image_uuid = uuid.UUID(image_uuid_str)
        except ValueError:
            image_uuid_invalid = True

        # Generate UUID
        if image_uuid_invalid:
            while image_uuid in image_uuids:
                image_uuid = uuid.uuid4()
        image_uuids.append(image_uuid)

        # Save image
        image_path = image_dir / f"{image_uuid}.jpeg"
        if not os.path.isfile(image_path):
            with open(image_path, "wb") as file:
                file.write(response.content)
                file.close()
        else:
            _raise_runtime_error(f"Image file '{image_path}' already exists.")

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
                f"Error retrieving from right hand image for record '{record_id}'. "
                f"Received HTTP status code {response.status_code}."
            )

        # Check if file name is a valid UUID
        image_uuid_str = os.path.splitext(record["right_hand_image_file"])[0]
        image_uuid_invalid = False
        try:
            image_uuid = uuid.UUID(image_uuid_str)
        except ValueError:
            image_uuid_invalid = True

        # Generate UUID
        if image_uuid_invalid:
            while image_uuid in image_uuids:
                image_uuid = uuid.uuid4()
        image_uuids.append(image_uuid)

        # Save image
        image_path = image_dir / f"{image_uuid}.jpeg"
        if not os.path.isfile(image_path):
            with open(image_path, "wb") as file:
                file.write(response.content)
                file.close()
        else:
            _raise_runtime_error(f"Image file '{image_path}' already exists.")


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

    metadata, invalid_data = _clean_metadata(raw_data)

    # Save metadata to file
    raw_data.to_csv(output_dir / "raw_data.csv", index=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    invalid_data.to_csv(output_dir / "invalid_data.csv", index=False)

    # Emit info message
    typer.echo(
        f"{len(raw_data)} records processed. "
        f"(valid: {len(metadata)}, invalid: {len(invalid_data)})"
    )
    typer.echo(f"Finished cleaning metadata in {time.time() - t_start:0.2f}s\n")

    # --- Retrieve images

    # Start timer
    t_start = time.time()

    # Download images
    _download_images(config, image_dir, metadata)

    # Emit info message
    typer.echo(f"Successfully retrieved images in {time.time() - t_start:0.2f}s\n")


# --- Run app

if __name__ == "__main__":
    typer.run(main)
