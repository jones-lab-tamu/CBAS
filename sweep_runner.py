r"""
CBAS Headless Hyperparameter Sweep Runner

This script allows a power-user to run a series of automated training experiments
("a sweep") to find the best hyperparameters for their specific dataset. It runs
in "headless" mode, meaning it does not require the GUI and can be left to run
overnight from the command line.

The script will iterate through all combinations of parameters defined in the
PARAMETER_GRID, train a model for each, and save the results to a single CSV file.

--- USAGE ---
1. Activate your Python virtual environment:
   (Windows): .\venv\Scripts\activate
   (macOS/Linux): source venv/bin/activate

2. Run the script from your main CBAS project directory, providing the required arguments:
   python sweep_runner.py --project_path "C:/Path/To/Your/CBAS_Project" --dataset_name "your_dataset_name"

3. The script will print its progress to the console and, when finished, will create a
   'sweep_results_[dataset_name]_[timestamp].csv' file in your project's root directory.
"""

import argparse
import itertools
import os
import sys
import time
from datetime import datetime

import pandas as pd
import torch
import yaml

# --- Add the backend directory to the Python path ---
# This is crucial to allow this script to import the CBAS backend modules.
script_dir = os.path.dirname(os.path.realpath(__file__))
backend_path = os.path.join(script_dir, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# --- Import CBAS backend modules ---
# These imports will now work because of the path modification above.
import cbas
import gui_state
import workthreads
from workthreads import TrainingTask

# ==============================================================================
# --- 1. EXPERIMENT DEFINITION (USER CONFIGURATION) ---
# This is the main section for a power-user to configure their experiment.
# ==============================================================================

# --- A. Define the number of times to REPLICATE each experiment ---
# This is the number of times the entire training process will be run for each
# unique combination of hyperparameters. Use a value >= 2 to measure variance.
REPLICATES_PER_SETTING = 3

# --- B. Define the parameters you want to VARY ---
# The script will test every possible combination of these values.
PARAMETER_GRID = {
    # Recommended sweep: [0.0, 0.0001, 0.001, 0.005]
    'weight_decay': [0.0, 0.0001, 0.001, 0.005],

    # Recommended sweep: [64, 128, 256]
    'lstm_hidden_size': [64, 128, 256],

    # Recommended sweep: [0.0, 0.1]
    'label_smoothing': [0.0, 0.1],

    # Recommended sweep: [1, 2]
    'lstm_layers': [1,2]
}

# --- C. Define the parameters you want to KEEP CONSTANT ---
# These settings will be used for all experimental runs.
FIXED_PARAMETERS = {
    # --- Core Training Setup ---
    'training_method': 'oversampling',
    'optimization_target': 'macro avg',
    'learning_rate': 0.0001,
    'epochs': 10
    'patience': 3,
    'batch_size': 1024,
    'sequence_length': 31,

    # --- Test Set Configuration ---
    'use_test': True,
    'test_split': 0.15,

    # --- Number of Runs ---
    # For a sweep, n=2 or n=3 is usually sufficient to get a directional signal.
    # Increase this for a final, rigorous evaluation of your single best model.
    'num_runs': 2 # This is the "inner" loop, finding the best model within one replicate.
    'num_trials': 2 
}

# ==============================================================================
# --- 2. HEADLESS CBAS INITIALIZATION ---
# This section performs the essential startup tasks that the GUI app normally handles.
# ==============================================================================

def initialize_headless_cbas(project_path: str):
    """
    Loads a CBAS project and initializes necessary components like the encoder.
    """
    print("--- Initializing Headless CBAS Environment ---")
    try:
        gui_state.proj = cbas.Project(project_path)
        print(f"Successfully loaded project: {gui_state.proj.path}")
    except cbas.InvalidProject as e:
        print(f"FATAL ERROR: {e}. The provided path is not a valid CBAS project.")
        sys.exit(1)

    # Initialize the DINOv2 encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        gui_state.dino_encoder = cbas.DinoEncoder(
            model_identifier=gui_state.proj.encoder_model_identifier,
            device=device
        )
        print(f"Successfully initialized encoder: {gui_state.proj.encoder_model_identifier}")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize the DINOv2 encoder. {e}")
        sys.exit(1)
    
    # We need a dummy TrainingThread object to call its methods, but we won't start it.
    gui_state.training_thread = workthreads.TrainingThread(device)
    print("-------------------------------------------\n")


# ==============================================================================
# --- 3. SWEEP EXECUTION AND REPORTING ---
# This section contains the main logic for running the sweep and saving results.
# ==============================================================================

def run_sweep(project_path: str, dataset_name: str):
    """
    Main function to execute the hyperparameter sweep with replications.
    """
    if dataset_name not in gui_state.proj.datasets:
        print(f"FATAL ERROR: Dataset '{dataset_name}' not found in project '{project_path}'.")
        sys.exit(1)

    keys, values = zip(*PARAMETER_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_jobs = len(param_combinations) * REPLICATES_PER_SETTING

    print(f"--- Starting Hyperparameter Sweep ---")
    print(f"Dataset: {dataset_name}")
    print(f"Found {len(param_combinations)} unique parameter combinations.")
    print(f"Running {REPLICATES_PER_SETTING} replicate(s) for each combination.")
    print(f"Total training jobs to execute: {total_jobs}")
    print("-------------------------------------\n")

    all_results = []
    job_counter = 0

    # The new outer loop for replications.
    for replicate_num in range(REPLICATES_PER_SETTING):
        print(f"========= STARTING REPLICATE {replicate_num + 1}/{REPLICATES_PER_SETTING} =========\n")
        
        for i, params in enumerate(param_combinations):
            job_counter += 1
            job_start_time = time.time()
            print(f"--- Starting Job {job_counter}/{total_jobs} (Replicate {replicate_num + 1}) ---")
            print(f"Parameters: {params}")

            current_params = FIXED_PARAMETERS.copy()
            current_params.update(params)

            task = TrainingTask(
                name=dataset_name,
                dataset=gui_state.proj.datasets[dataset_name],
                behaviors=gui_state.proj.datasets[dataset_name].config.get('behaviors', []),
                custom_weights=None,
                **current_params
            )

            gui_state.training_thread._execute_training_task(task)
            
            report_path = os.path.join(gui_state.proj.datasets[dataset_name].path, "performance_report.yaml")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = yaml.safe_load(f)
                
                run_reports = report_data.get('run_results', [])
                if run_reports:
                    result_row = current_params.copy()
                    result_row.update(params)
                    
                    # Add the replicate number to the results row for easy analysis.
                    result_row['replicate'] = replicate_num + 1
                    
                    test_f1_scores = [r['test_report'].get('macro avg', {}).get('f1-score', 0) for r in run_reports]
                    avg_test_f1 = sum(test_f1_scores) / len(test_f1_scores) if test_f1_scores else 0
                    result_row['avg_test_f1_macro'] = avg_test_f1
                    
                    best_run_idx = int(pd.Series([r['validation_report'].get('macro avg', {}).get('f1-score', 0) for r in run_reports]).idxmax())
                    best_run_report = run_reports[best_run_idx]

                    for behavior in task.behaviors:
                        f1_score = best_run_report['test_report'].get(behavior, {}).get('f1-score', 0)
                        result_row[f'{behavior}_Test_F1'] = f1_score

                    all_results.append(result_row)
            
            job_end_time = time.time()
            print(f"--- Finished Job {job_counter}/{total_jobs} in {job_end_time - job_start_time:.2f} seconds ---\n")

    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sweep_results_{dataset_name}_{timestamp}.csv"
        output_path = os.path.join(gui_state.proj.path, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"--- SWEEP COMPLETE ---")
        print(f"Results saved to: {output_path}")
        print("----------------------")
    else:
        print("--- SWEEP COMPLETE ---")
        print("No results were generated. Please check logs for errors.")
        print("----------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBAS Headless Hyperparameter Sweep Runner")
    parser.add_argument('--project_path', required=True, type=str, help="The absolute path to the CBAS project directory.")
    parser.add_argument('--dataset_name', required=True, type=str, help="The name of the dataset to run the sweep on.")
    args = parser.parse_args()

    initialize_headless_cbas(args.project_path)
    run_sweep(args.project_path, args.dataset_name)