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
import os
import typer
import csv
import logging
import datetime
from pathlib import Path

# External packages
import dermaml.model_setup as model_setup
from pycaret import regression
import yaml

# --- Main program
# FIXME write results to file with date

def main(
        config_file: Path=typer.Argument(
            "/Users/ntin/DermaML/bin/config.yaml", help="Path to the YAML configuration file."
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    now = datetime.datetime.today().strftime('%Y-%m-%d %H-%M')
    
    # read configuration files
    config = model_setup.read_config_yaml(config_file)
    metadata_target = config['metadata_target']

    # set saving location
    output_loc = config.get('paths', {})['output_dir']
    metadata_used = config.get('paths', {})['metadata_file'].split('/')[-1]
    folder_name = '{}_automl_{}'.format(now, metadata_used)
    output_dir = os.path.join(output_loc, folder_name)
    os.mkdir(output_dir)

    # prepare saving files
    save_best_models = os.path.join(output_dir, 'automl-best.yaml')
    save_results = os.path.join(output_dir, 'automl-scores.csv')

    # prepare dataset
    train, test = model_setup.tabular_input(config)
    logging.info(f'using metadata: {metadata_used}')
    # logging.info(f'train contents: {train.columns}')
    # logging.info(f'test set indices: {test.index}')
    logging.info(f'train size: {train.shape}, test size: {test.shape}')


    # --- Perform AutoML evaluation

    # Set up the dataset for AutoML
    regression.setup(
        data=train,
        test_data=test,
        fold_strategy='kfold',
        target=metadata_target,
        experiment_name=experiment_name,
        html=False,
        # log_experiment=True,
        # silent=True,
        verbose=True
    )

    # Automatically train, test, and evaluate models
    best_models = regression.compare_models(
        n_select=num_best,
        verbose=False
    )

    # --- Save results

    # Best models
    best_models = [' '.join(s.strip() for s in str(model).split('\n'))
                   for model in best_models]
    with open(save_best_models, 'w') as file:
        yaml.dump(best_models, file, width=float("inf"))

    # Model scores
    regression.pull().to_csv(
        save_results, 
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )
    logging.info(f'Saved to {save_results}')


# --- Run app

if __name__ == "__main__":
    typer.run(main)
