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
# FIXME read in filenames from yaml
# FIXME implement in typer

def not_main(
        configuration_file: Path,
         best_models_file: Path = typer.Option("automl-best.yaml",
                                               "-m", "--models"),
         scores_file: Path = typer.Option("automl-scores.csv",
                                          "-s", "--scores"),
         num_best: int = 5,
         experiment_name: str = "automl",
         ) -> None:
    """
    Run AutoML evaluation.

    Results are output two files: 'model-scores.csv'
    """
    # --- Check arguments
    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations
    # read configuration files
    yaml.SafeLoader.add_constructor('!join', model_setup.join_constructor)
    with open(configuration_file, 'r') as f:
        config = yaml.safe_load(f)
    feature_file = config['tabular_feature_file']
    metadata_file = config['metadata_file']
    metadata_target = config['metadata_target']


    X = model_setup.tabular_input(
        config_file=configuration_file,
    )

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
    typer.run(not_main)
