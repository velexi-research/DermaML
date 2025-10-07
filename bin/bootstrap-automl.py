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
import random
import datetime
from pathlib import Path

# External packages
import dermaml.model_setup as model_setup
from pycaret import regression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Main program
# FIXME write results to file with date

def main(
        config_file: Path=typer.Argument(
            "/Users/ntin/DermaML/bin/config.yaml", help="Path to the YAML configuration file."
        ),
        iters: int = typer.Option(
            100, help="Number of models to generate to bootstrap from."
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
    if iters <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    now = datetime.datetime.today().strftime('%Y-%m-%d %H-%M')
    
    # read configuration files
    config = model_setup.read_config_yaml(config_file)
    metadata_target = config['metadata_target']
    print('HELLOO')

    # set saving location
    output_loc = config.get('paths', {})['output_dir']
    metadata_used = config.get('paths', {})['metadata_file'].split('/')[-1]
    folder_name = '{}_automl_{}'.format(now, metadata_used)
    output_dir = os.path.join(output_loc, folder_name)
    # output_dir = output_loc
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # prepare saving files
    # save_best_models = os.path.join(output_dir, 'automl-best.yaml')
    save_results = os.path.join(output_dir, 'automl-scores.csv')
    save_predictions = os.path.join(output_dir, 'automl-predictions.csv')
    scores = pd.DataFrame()
    predictions = pd.DataFrame(columns=['iter', 'predicted', 'actual'])

    for i in range(iters):
        # prepare dataset
        Xy, _train, _test = model_setup.tabular_input(config, random_state=random.randint(0,500))
        train, test = _train.drop(columns=['metadata_instance_id']), _test.drop(columns=['metadata_instance_id'])
        logging.info(f'using metadata: {metadata_used}')
        logging.info(f'train size: {train.shape}, test size: {test.shape}')
        

        # --- Perform AutoML evaluation
        # Set up the dataset for AutoML
        regression.setup(
            data=train,
            test_data=test,
            target=metadata_target,
            experiment_name=experiment_name,
        )

        # Automatically train, test, and evaluate models
        best_models = regression.compare_models(sort='mae')
        prediction_table = regression.predict_model(best_models, data=test)

        # Predictions

        predictions = pd.concat(
            [predictions, 
             pd.DataFrame({
            "iter":i,
            "predicted": prediction_table['prediction_label'],
            'actual':test[metadata_target],
            'metadata_ID':_test['metadata_instance_id'],})
            ],
            ignore_index=True
        )
        
        # Model scores
        scores = pd.concat([scores, regression.pull()])
    
    predictions.to_csv(save_predictions)
    pd.DataFrame(scores).to_csv(
            save_results, 
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )
    
    # --- Plot residuals
    sns.set_context('paper')
    sns.set_theme()
    sns.scatterplot(data=predictions, x='actual', y='predicted', 
                    hue='iter', style='iter')
    plt.savefig(os.path.join(output_dir, 'residuals.png'))

    logging.info(f'Saved to {save_results}')

    
# --- Run app

if __name__ == "__main__":
    typer.run(main)
