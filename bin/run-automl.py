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
Script for running AutoML evaluation.
"""

# --- Imports

# Standard library
import csv
from pathlib import Path

# External packages
import model_setup
from pycaret import regression
import typer
import yaml


# --- Main program
# FIXME write results to file with date

def main(
        config_file: Path = typer.Argument(..., help="Path to the YAML configuration file."),
        best_models_file: Path = typer.Option(
            "automl-best.yaml", "-m", "--models", help="Path to save the best models as a YAML file."
        ),
        scores_file: Path = typer.Option(
            "automl-scores.csv", "-s", "--scores", help="Path to save model performance scores as a CSV file."
        ),
        num_best: int = typer.Option(
            5, help="Number of top models to select during AutoML evaluation. Must be strictly positive."
        ),
        experiment_name: str = typer.Option(
            "automl", help="Name for the AutoML experiment (for logging purposes)."
        ),
    ) -> None:
    """
    Runs AutoML evaluation on a tabular dataset and outputs the best models and their performance scores.

    This function:
    - Loads a YAML configuration file specifying input data and target metadata.
    - Sets up an AutoML environment for regression.
    - Selects the top `num_best` models based on performance.
    - Saves the best models to a YAML file and performance scores to a CSV file.

    Args:
        config_file (Path): Path to the YAML configuration file.
        best_models_file (Path): Path to save the best models (YAML).
        scores_file (Path): Path to save model performance scores (CSV).
        num_best (int): Number of top models to select (must be strictly positive).
        experiment_name (str): Name of the AutoML experiment.

    Raises:
        typer.Abort: If `num_best` is not strictly positive.

    Output:
        Results are output two files: 'model-scores.csv' and 'automl-best.yaml'

    Generated with an LLM 2025 March 13.
    """
    # --- Check arguments
    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations
    
    # read configuration files
    config = model_setup.read_config_yaml(config_file)
    metadata_target = config['metadata_target']
    X = model_setup.tabular_input(config)

    # --- Perform AutoML evaluation

    # Set up the dataset for AutoML
    regression.setup(data=X,
                         target=metadata_target,
                         log_experiment=True,
                         experiment_name=experiment_name,
                         html=False,
                         silent=True,
                         verbose=False)

    # Automatically train, test, and evaluate models
    best_models = regression.compare_models(n_select=num_best,
                                                verbose=False)

    # --- Save results

    # Best models
    best_models = [' '.join(s.strip() for s in str(model).split('\n'))
                   for model in best_models]
    with open(best_models_file, 'w') as file_:
        yaml.dump(best_models, file_, width=float("inf"))

    # Model scores
    regression.pull().to_csv(scores_file, index=False,
                                 quoting=csv.QUOTE_NONNUMERIC)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
