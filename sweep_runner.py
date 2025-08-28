r"""
CBAS Headless Hyperparameter Sweep Runner

This script allows a power-user to run a series of automated training experiments
("a sweep") to find the best hyperparameters for their specific dataset. It runs
in "headless" mode, meaning it does not require the GUI and can be left to run
overnight from the command line.

The script will iterate through all combinations of parameters defined in the
PARAMETER_GRID, train a model for each, and save the results to a single CSV file.

--- USAGE ---
The script is designed to be run in sequential phases.

1. Activate your Python virtual environment:
   (Windows): .\venv\Scripts\activate
   (macOS/Linux): source venv/bin/activate

2. Run the script from your main CBAS project directory, providing the required arguments for each phase:

   # PHASE 0: Pre-compute the data splits (Run this ONCE per experiment)
   # This creates the sweep_splits.json and outer_splits.json files.
   python sweep_runner.py --project_path "C:/Path/To/Your/CBAS_Project" --dataset_name "your_dataset_name" --phase precompute

   # PHASE 1: Run the hyperparameter sweep (This is the long, overnight run)
   # This uses sweep_splits.json to test all parameter combinations and outputs sweep_results.csv.
   python sweep_runner.py --project_path "C:/Path/To/Your/CBAS_Project" --dataset_name "your_dataset_name" --phase sweep

   # PHASE 2: Run the final, rigorous evaluation of the champion model
   # After analyzing sweep_results.csv, you will update the CHAMPION_PARAMETERS in this script.
   # This phase then uses outer_splits.json to generate the final, publishable performance numbers.
   python sweep_runner.py --project_path "C:/Path/To/Your/CBAS_Project" --dataset_name "your_dataset_name" --phase evaluate

   # PHASE 3: Train the single, deployable model for future use
   # This trains one final model on a large portion of the data using the champion hyperparameters.
   python sweep_runner.py --project_path "C:/Path/To/Your/CBAS_Project" --dataset_name "your_dataset_name" --phase train_final

3. The script will print its progress to the console and create output files
   (splits manifests, results CSVs) in your project's root directory.
"""

import argparse
import itertools
import os
import sys
import time
from datetime import datetime
import json
import hashlib
from collections import defaultdict
import random

import pandas as pd
import torch
import yaml
import numpy as np

# --- Add the backend directory to the Python path ---
# This is crucial to allow this script to import the CBAS backend modules.
script_dir = os.path.dirname(os.path.realpath(__file__))
backend_path = os.path.join(script_dir, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# --- Import CBAS backend modules ---
import cbas
import gui_state
import workthreads
from workthreads import TrainingTask
from backend.splits import RandomSplitProvider, ManifestSplitProvider, _generate_dataset_fingerprint

# ==============================================================================
# --- 1. EXPERIMENT DEFINITION (USER CONFIGURATION) ---
# ==============================================================================

# --- A. Define the parameters you want to VARY in the sweep ---
PARAMETER_GRID = {
    'weight_decay': [0.0001, 0.001],
    'lstm_hidden_size': [64, 128],
    'label_smoothing': [0.0, 0.1],
    'lstm_layers': [1, 2]
}

# --- B. Define the parameters you want to KEEP CONSTANT for the sweep ---
SWEEP_FIXED_PARAMETERS = {
    'training_method': 'oversampling',
    'optimization_target': 'macro avg',
    'learning_rate': 0.0001,
    'epochs': 10,
    'patience': 3,
    'batch_size': 1024,
    'sequence_length': 31,
    'use_test': False, # Test set is NOT used in the sweep phase
    'test_split': 0.0,
    'num_runs': 5, # 5 replicates for the sweep
    'num_trials': 2
}

# --- C. Define the CHAMPION parameters for final evaluation & training ---
# After running the sweep, you will manually update these with the best combination.
# This dictionary is now self-contained and defines all necessary parameters.
CHAMPION_PARAMETERS = {
    # --- Champion Hyperparameters (Update these after your sweep) ---
    'weight_decay': 0.0001,
    'lstm_hidden_size': 128,
    'label_smoothing': 0.1,
    'lstm_layers': 2,

    # --- Fixed Experimental Conditions ---
    'training_method': 'oversampling',
    'optimization_target': 'macro avg',
    'learning_rate': 0.0001,
    'epochs': 10,
    'patience': 3,
    'batch_size': 1024,
    'sequence_length': 31,
    
    # --- Final Evaluation Settings ---
    'use_test': True,
    'test_split': 0.15,
    'num_runs': 20,
    'num_trials': 2  # Trials per run can be kept low as we trust the params
}

# ==============================================================================
# --- 2. HEADLESS CBAS INITIALIZATION ---
# ==============================================================================

def initialize_headless_cbas(project_path: str):
    """
    Loads a CBAS project and initializes necessary components like the encoder.
    """
    # Set the required environment variable for deterministic CuBLAS operations.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    gui_state.HEADLESS_MODE = True
    print("--- Initializing Headless CBAS Environment ---")
    try:
        gui_state.proj = cbas.Project(project_path)
        print(f"Successfully loaded project: {gui_state.proj.path}")
    except cbas.InvalidProject as e:
        print(f"FATAL ERROR: {e}. The provided path is not a valid CBAS project.")
        sys.exit(1)

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
    
    gui_state.training_thread = workthreads.TrainingThread(device)
    
    # Set full determinism for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Set all random seeds and enabled deterministic algorithms.")
    print("-------------------------------------------\n")

# ==============================================================================
# --- 3. SWEEP PHASES ---
# ==============================================================================

def precompute_splits(dataset_name: str):
    """
    Generates and saves two manifest files with stratified, group-aware splits.
    """
    print("--- Phase: Pre-computing Splits ---")
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    print(f"Dataset Fingerprint: {fingerprint}")

    all_instances = [inst for b in dataset.config.get("behaviors", []) for inst in dataset.labels.get("labels", {}).get(b, [])]
    all_subjects = list(set(os.path.dirname(inst['video']) for inst in all_instances))

    # Generate splits for the sweep (Phase 1)
    sweep_manifest = {
        "manifest_type": "hyperparameter_sweep",
        "dataset_fingerprint": fingerprint,
        "splits": []
    }
    provider = RandomSplitProvider(split_ratios=(0.85, 0.15, 0.0)) # 85% train, 15% val, 0% test
    for i in range(10): # Generate 10 splits for the fine-grained sweep
        train, val, test = provider.get_split(i, all_subjects, all_instances, dataset.config.get("behaviors", []))
        sweep_manifest["splits"].append({"train": train, "validation": val, "test": test})
    
    sweep_path = os.path.join(gui_state.proj.path, "sweep_splits.json")
    with open(sweep_path, 'w') as f:
        json.dump(sweep_manifest, f, indent=4)
    print(f"Successfully saved sweep splits to: {sweep_path}")

    # Generate splits for the final evaluation (Phase 2)
    outer_manifest = {
        "manifest_type": "outer_evaluation",
        "dataset_fingerprint": fingerprint,
        "splits": []
    }
    provider = RandomSplitProvider(split_ratios=(0.70, 0.15, 0.15)) # 70/15/15 split
    for i in range(20): # Generate 20 outer splits
        train, val, test = provider.get_split(i, all_subjects, all_instances, dataset.config.get("behaviors", []))
        outer_manifest["splits"].append({"train": train, "validation": val, "test": test})

    outer_path = os.path.join(gui_state.proj.path, "outer_splits.json")
    with open(outer_path, 'w') as f:
        json.dump(outer_manifest, f, indent=4)
    print(f"Successfully saved outer evaluation splits to: {outer_path}")
    print("-------------------------------------\n")


def run_sweep(dataset_name: str):
    """
    Main function to execute the hyperparameter sweep using a manifest.
    """
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    manifest_path = os.path.join(gui_state.proj.path, "sweep_splits.json")

    keys, values = zip(*PARAMETER_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_jobs = len(param_combinations)
    print(f"--- Phase: Hyperparameter Sweep ---")
    print(f"Found {total_jobs} unique parameter combinations to test.")
    print(f"Each combination will be run {SWEEP_FIXED_PARAMETERS['num_runs']} times using pre-computed splits.")
    print("-------------------------------------\n")

    all_results = []
    
    for i, params in enumerate(param_combinations):
        job_start_time = time.time()
        print(f"--- Starting Job {i+1}/{total_jobs} ---")
        print(f"Parameters: {params}")

        current_params = SWEEP_FIXED_PARAMETERS.copy()
        current_params.update(params)

        task = TrainingTask(name=dataset_name, dataset=dataset, behaviors=dataset.config.get('behaviors', []), **current_params)
        
        # Instantiate the manifest provider for this job
        split_provider = ManifestSplitProvider(manifest_path, fingerprint)
        
        # Execute the training task by calling it directly
        gui_state.training_thread._execute_training_task(task, split_provider)
        
        report_path = os.path.join(dataset.path, "performance_report.yaml")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_data = yaml.safe_load(f)
            
            run_reports = report_data.get('run_results', [])
            if run_reports:
                result_row = current_params.copy()
                result_row.update(params)
                
                val_f1_scores = [r['validation_report'].get('macro avg', {}).get('f1-score', 0) for r in run_reports]
                avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores) if val_f1_scores else 0
                result_row['avg_validation_f1_macro'] = avg_val_f1
                all_results.append(result_row)
        
        job_end_time = time.time()
        print(f"--- Finished Job {i+1}/{total_jobs} in {job_end_time - job_start_time:.2f} seconds ---\n")

    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sweep_results_{dataset_name}_{timestamp}.csv"
        output_path = os.path.join(gui_state.proj.path, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"--- SWEEP COMPLETE ---")
        print(f"Results saved to: {output_path}")
    else:
        print("--- SWEEP COMPLETE ---")
        print("No results were generated.")


def run_final_evaluation(dataset_name: str):
    """
    Runs the final, rigorous evaluation on the champion model.
    """
    print("--- Phase: Final Evaluation ---")
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    manifest_path = os.path.join(gui_state.proj.path, "outer_splits.json")
    
    with open(manifest_path, 'r') as f:
        num_replicates = len(json.load(f)['splits'])
    
    print(f"Found {num_replicates} replicates in outer_splits.json.")
    print(f"Using Champion Hyperparameters: {CHAMPION_PARAMETERS}")
    
    # 1. Instantiate the provider ONCE, without the replicate_index.
    split_provider = ManifestSplitProvider(manifest_path, fingerprint)
    
    # 2. Copy champion parameters and align num_runs to the manifest length.
    #    This prevents out-of-range indices if the manifest has a non-20 count.
    eval_params = CHAMPION_PARAMETERS.copy()
    eval_params['num_runs'] = num_replicates
    task = TrainingTask(name=dataset_name,
                        dataset=dataset,
                        behaviors=dataset.config.get('behaviors', []),
                        **eval_params)
    
    # 3. Call the training engine ONCE. The engine will iterate over range(task.num_runs),
    #    calling provider.get_split(0), get_split(1), ... up to num_replicates - 1.
    job_start_time = time.time()
    gui_state.training_thread._execute_training_task(task, split_provider)
    job_end_time = time.time()
    print(f"--- Finished Final Evaluation in {job_end_time - job_start_time:.2f} seconds ---\n")

    all_results = []
    report_path = os.path.join(dataset.path, "performance_report.yaml")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_data = yaml.safe_load(f)
        
        # The report will now contain results for all 20 runs.
        run_reports = report_data.get('run_results', [])
        for i, run_report in enumerate(run_reports):
            result_row = CHAMPION_PARAMETERS.copy()
            result_row['replicate'] = i + 1
            
            for behavior in task.behaviors:
                f1_score = run_report['test_report'].get(behavior, {}).get('f1-score', 0)
                result_row[f'{behavior}_Test_F1'] = f1_score
            
            macro_f1 = run_report['test_report'].get('macro avg', {}).get('f1-score', 0)
            result_row['avg_test_f1_macro'] = macro_f1
            all_results.append(result_row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"final_evaluation_results_{dataset_name}_{timestamp}.csv"
        output_path = os.path.join(gui_state.proj.path, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"--- FINAL EVALUATION COMPLETE ---")
        print(f"Results saved to: {output_path}")
    else:
        print("--- FINAL EVALUATION COMPLETE ---")
        print("No results were generated.")


def train_final_model(dataset_name: str):
    """
    Trains one final, deployable model on a representative split using the
    champion hyperparameters.
    """
    print("--- Phase: Training Final Deployable Model ---")
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    manifest_path = os.path.join(gui_state.proj.path, "outer_splits.json")

    print(f"Using Champion Hyperparameters: {CHAMPION_PARAMETERS}")
    
    # 1. Load the manifest and select one representative split (the first one)
    print("Loading representative split from outer_splits.json (replicate 0)")
    manifest_provider = ManifestSplitProvider(manifest_path, fingerprint)
    train_subjects, val_subjects, _ = manifest_provider.get_split(0, [], [], []) # Test set is ignored

    # 2. Combine train and validation subjects into one large training pool
    final_train_subjects = train_subjects + val_subjects
    print(f"Combining train and validation sets into a final training pool of {len(final_train_subjects)} subjects.")

    # 3. Create a new Task. We will not use a validation or test set for this final fit.
    # We set num_runs and num_trials to 1 as we are only producing one artifact.
    final_params = CHAMPION_PARAMETERS.copy()
    final_params['num_runs'] = 1
    final_params['num_trials'] = 1
    final_params['use_test'] = False # Explicitly do not use a test set
    final_params['test_split'] = 0.0
    
    task = TrainingTask(name=dataset_name, dataset=dataset, behaviors=dataset.config.get('behaviors', []), **final_params)

    # 4. Create a custom SplitProvider that returns our combined list and no validation set.
    class FinalFitSplitProvider(cbas.SplitProvider):
        def __init__(self, final_train_subjects):
            self.final_train_subjects = final_train_subjects
        def get_split(self, run_index, all_subjects, all_instances, behaviors):
            # Return the combined list for training, and empty lists for val/test
            return self.final_train_subjects, [], []

    final_split_provider = FinalFitSplitProvider(final_train_subjects)

    # 5. Execute the training job
    print("Starting final training job...")
    job_start_time = time.time()
    
    # The _execute_training_task will now run, but the val_ds will be empty,
    # so early stopping based on validation performance will not trigger.
    # The model will train for the full number of epochs specified.
    gui_state.training_thread._execute_training_task(task, final_split_provider)
    
    job_end_time = time.time()
    print(f"--- Finished Final Training in {job_end_time - job_start_time:.2f} seconds ---")
    print(f"The final, deployable model has been saved to the project's 'models' directory.")
    print("The official performance of this model is the one reported from the 'evaluate' phase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBAS Headless Experimental Runner")
    parser.add_argument('--project_path', required=True, type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--phase', required=True, type=str, choices=['precompute', 'sweep', 'evaluate', 'train_final'])
    args = parser.parse_args()

    initialize_headless_cbas(args.project_path)
    
    if args.phase == 'precompute':
        precompute_splits(args.dataset_name)
    elif args.phase == 'sweep':
        run_sweep(args.dataset_name)
    elif args.phase == 'evaluate':
        run_final_evaluation(args.dataset_name)
    elif args.phase == 'train_final':
        train_final_model(args.dataset_name)