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
import glob
import traceback

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
from backend.splits import RandomSplitProvider, ManifestSplitProvider, _generate_dataset_fingerprint, SplitProvider

# ==============================================================================
# --- 0. HELPER FUNCTIONS ---
# ==============================================================================

def _nice_multiple(x, base=32, minimum=32):
    """
    Snap x down to the nearest multiple of 'base', not below 'minimum'.
    """
    return max(minimum, (x // base) * base)

def derive_batch_size_for_seq_len(seq_len: int,
                                  base_batch: int = 1024,
                                  base_seq: int = 31,
                                  batch_cap: int = 1024,
                                  minimum: int = 32,
                                  snap: int = 32) -> int:
    """
    Keep batch_size * seq_len approximately constant relative to a baseline.
    """
    target_tokens = base_batch * base_seq
    raw = max(1, target_tokens // max(1, seq_len))
    bs = _nice_multiple(raw, base=snap, minimum=minimum)
    # Cap at the user-defined base_batch (e.g. 1024)
    return min(bs, batch_cap)

# ==============================================================================
# --- 1. EXPERIMENT DEFINITION (USER CONFIGURATION) ---
# ==============================================================================

# --- A. Define the parameters you want to VARY in the sweep ---
PARAMETER_GRID = {
    'weight_decay': [1e-4, 2e-4], # default [0.0001, 0.001]
    'lstm_hidden_size': [128], # default [64, 128]
    'label_smoothing': [0.1], # default [0.0, 0.1]
    'lstm_layers': [1], #default [1, 2]
    'learning_rate': [5e-5, 7e-5], # default [5e-5, 1e-4, 3e-4]
    'sequence_length': [63, 95] # default [31, 63] - MUST BE ODD
}

# --- B. Define the parameters you want to KEEP CONSTANT for the sweep ---
SWEEP_FIXED_PARAMETERS = {
    'training_method': 'oversampling',
    'optimization_target': 'weighted avg', # Changed to weighted avg as it's safer for unbalanced data
    'epochs': 10,
    'patience': 3,
    'batch_size': 1024, # This serves as the BASE reference for the budget calculation
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
    'weight_decay': 1e-4,
    'lstm_hidden_size': 128,
    'label_smoothing': 0.1,
    'lstm_layers': 1,

    # --- Fixed Experimental Conditions ---
    'training_method': 'oversampling',
    'optimization_target': 'weighted avg',
    'learning_rate': 5e-5,
    'epochs': 10,
    'patience': 3,
    'batch_size': 1024,
    'sequence_length': 63, # MUST BE ODD
    
    # --- Final Evaluation Settings ---
    'use_test': True,
    'test_split': 0.15,
    'num_runs': 15,
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
    for i in range(10): 
        # Allow fallback
        train, val, test = provider.get_split(
            i, all_subjects, all_instances, dataset.config.get("behaviors", []), 
            allow_relaxed_fallback=True
        )
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
    for i in range(20): 
        # Allow fallback
        train, val, test = provider.get_split(
            i, all_subjects, all_instances, dataset.config.get("behaviors", []), 
            allow_relaxed_fallback=True
        )
        outer_manifest["splits"].append({"train": train, "validation": val, "test": test})

    outer_path = os.path.join(gui_state.proj.path, "outer_splits.json")
    with open(outer_path, 'w') as f:
        json.dump(outer_manifest, f, indent=4)
    print(f"Successfully saved outer evaluation splits to: {outer_path}")
    print("-------------------------------------\n")


def run_sweep(dataset_name: str):
    """
    Execute the hyperparameter sweep using a precomputed manifest of splits.
    """
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    manifest_path = os.path.join(gui_state.proj.path, "sweep_splits.json")
    experiments_dir = os.path.join(dataset.path, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    # Build all parameter combinations from the grid
    keys, values = zip(*PARAMETER_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_jobs = len(param_combinations)
    print("--- Phase: Hyperparameter Sweep ---")
    print(f"Found {total_jobs} unique parameter combinations to test.")
    print(f"Each combination will be run {SWEEP_FIXED_PARAMETERS['num_runs']} times using pre-computed splits.")
    print("-------------------------------------\n")

    all_results = []
    base_seq = SWEEP_FIXED_PARAMETERS.get('sequence_length', 31)
    base_batch = SWEEP_FIXED_PARAMETERS.get('batch_size', 1024)

    for i, params in enumerate(param_combinations):
        job_start_time = time.time()

        # Compose parameters: start from fixed, then overlay grid
        current_params = SWEEP_FIXED_PARAMETERS.copy()
        current_params.update(params)

        # Dynamically derive the batch size to keep token count constant
        # Allow up to the user-defined base_batch_size
        base_batch_size = SWEEP_FIXED_PARAMETERS.get('batch_size', 1024)
        current_params['batch_size'] = derive_batch_size_for_seq_len(
            seq_len=current_params['sequence_length'],
            base_batch=base_batch_size,
            base_seq=31,
            batch_cap=base_batch_size # Uses the user-defined limit from FIXED_PARAMETERS
        )
        print(f"--- Starting Job {i+1}/{total_jobs} ---")
        # Log both the grid parameters and the final effective parameters for clarity
        print(f"Parameters (grid): {params}")
        print(f"Parameters (effective): {current_params}")
        print(f"Derived Batch Size: {current_params['batch_size']}")

        task = TrainingTask(name=dataset_name, dataset=dataset, behaviors=dataset.config.get('behaviors', []), **current_params)
        split_provider = ManifestSplitProvider(manifest_path, fingerprint)

        # Unique output directory for this job's artifacts
        param_str = "_".join([f"{k.replace('_','-')}-{v}" for k, v in params.items()])
        job_output_dir = os.path.join(experiments_dir, f"sweep_{param_str}")

        gui_state.training_thread._execute_training_task(
            task, split_provider, output_dir=job_output_dir, plot_suffix='runs'
        )

        # Collect results
        report_path = os.path.join(job_output_dir, "performance_report.yaml")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_data = yaml.safe_load(f)

            run_reports = report_data.get('run_results', [])
            if run_reports:
                result_row = current_params.copy()
                # Add comparability metadata
                result_row['effective_tokens_per_step'] = (
                    current_params['batch_size'] * current_params['sequence_length']
                )

                # Dynamically select the metric based on what was actually optimized
                target_metric = current_params.get('optimization_target', 'weighted avg')
                
                # Average validation F1 scores across runs using the correct metric
                val_f1_scores = [
                    r.get('validation_report', {}).get(target_metric, {}).get('f1-score', 0.0)
                    for r in run_reports
                ]
                avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores) if val_f1_scores else 0.0
                
                # Save as a distinct column so user knows which metric was used
                result_row[f'avg_validation_f1_{target_metric.replace(" ", "_")}'] = avg_val_f1

                all_results.append(result_row)

        job_end_time = time.time()
        print(f"--- Finished Job {i+1}/{total_jobs} in {job_end_time - job_start_time:.2f} seconds ---\n")

    # Persist sweep table
    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sweep_results_{dataset_name}_{timestamp}.csv"
        output_path = os.path.join(experiments_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print("--- SWEEP COMPLETE ---")
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
    experiments_dir = os.path.join(dataset.path, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    with open(manifest_path, 'r') as f:
        num_replicates = len(json.load(f)['splits'])
    
    print(f"Found {num_replicates} replicates in outer_splits.json.")
    print(f"Using Champion Hyperparameters: {CHAMPION_PARAMETERS}")
    
    all_results = []
    
    split_provider = ManifestSplitProvider(manifest_path, fingerprint)
    
    task = TrainingTask(name=dataset_name, dataset=dataset, behaviors=dataset.config.get('behaviors', []), **CHAMPION_PARAMETERS)
    
    eval_output_dir = os.path.join(experiments_dir, f"final_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    job_start_time = time.time()
    gui_state.training_thread._execute_training_task(task, split_provider, output_dir=eval_output_dir, plot_suffix='replicates')
    job_end_time = time.time()
    print(f"--- Finished Final Evaluation in {job_end_time - job_start_time:.2f} seconds ---\n")

    report_path = os.path.join(eval_output_dir, "performance_report.yaml")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_data = yaml.safe_load(f)
        
        run_reports = report_data.get('run_results', [])
        for i, run_report in enumerate(run_reports):
            result_row = CHAMPION_PARAMETERS.copy()
            result_row['replicate'] = i + 1
            
            for behavior in task.behaviors:
                # Extract all three metrics from the test report
                test_metrics = run_report['test_report'].get(behavior, {})
                f1_score = test_metrics.get('f1-score', 0)
                precision = test_metrics.get('precision', 0)
                recall = test_metrics.get('recall', 0)
                
                # Save each to a unique column
                result_row[f'{behavior}_Test_F1'] = f1_score
                result_row[f'{behavior}_Test_Precision'] = precision
                result_row[f'{behavior}_Test_Recall'] = recall
            
            # Also capture the target metric for the test set
            target_metric = CHAMPION_PARAMETERS.get('optimization_target', 'weighted avg')
            macro_f1 = run_report['test_report'].get(target_metric, {}).get('f1-score', 0)
            result_row[f'avg_test_f1_{target_metric.replace(" ", "_")}'] = macro_f1
            all_results.append(result_row)

    if all_results:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"final_evaluation_results_{dataset_name}_{timestamp}.csv"
        output_path = os.path.join(experiments_dir, output_filename)
        results_df.to_csv(output_path, index=False)
        print(f"--- FINAL EVALUATION COMPLETE ---")
        print(f"Results saved to: {output_path}")
    else:
        print("--- FINAL EVALUATION COMPLETE ---")
        print("No results were generated.")


def train_final_model(dataset_name: str):
    """
    Trains one final, deployable model and updates the GUI card with the
    rigorous performance metrics from the 'evaluate' phase.
    """
    print("--- Phase: Training Final Deployable Model ---")
    dataset = gui_state.proj.datasets[dataset_name]
    fingerprint = _generate_dataset_fingerprint(dataset)
    manifest_path = os.path.join(gui_state.proj.path, "outer_splits.json")
    experiments_dir = os.path.join(dataset.path, "experiments")

    print(f"Using Champion Hyperparameters: {CHAMPION_PARAMETERS}")
    
    # --- Part 1: Train and save the final model artifact ---
    manifest_provider = ManifestSplitProvider(manifest_path, fingerprint)
    # Get all subjects for this split, as we'll need them for counting later
    train_subjects, val_subjects, test_subjects = manifest_provider.get_split(0, [], [], [])

    final_train_subjects = train_subjects + val_subjects
    print(f"Combining train and validation sets into a final training pool of {len(final_train_subjects)} subjects.")

    final_params = CHAMPION_PARAMETERS.copy()
    final_params['num_runs'] = 1
    final_params['num_trials'] = 1
    final_params['use_test'] = False
    final_params['test_split'] = 0.0
    
    task = TrainingTask(name=dataset_name, dataset=dataset, behaviors=dataset.config.get('behaviors', []), **final_params)

    class FinalFitSplitProvider(SplitProvider):
        def __init__(self, final_train_subjects):
            self.final_train_subjects = final_train_subjects
        
        # Accept argument to satisfy TrainingThread interface
        def get_split(self, run_index, all_subjects, all_instances, behaviors, allow_relaxed_fallback=False):
            return self.final_train_subjects, [], []

    final_split_provider = FinalFitSplitProvider(final_train_subjects)

    print("Starting final training job to produce model.pth...")
    job_start_time = time.time()
    
    temp_output_dir = os.path.join(experiments_dir, f"final_train_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    gui_state.training_thread._execute_training_task(task, final_split_provider, output_dir=temp_output_dir)
    
    job_end_time = time.time()
    print(f"--- Finished Final Training in {job_end_time - job_start_time:.2f} seconds ---")
    print(f"The final, deployable model has been saved to the project's 'models' directory.")

    # --- Part 2: Update the main config.yaml with metrics and counts ---
    print("\n--- Updating GUI Card with Final Evaluation Metrics and Instance Counts ---")
    try:
        # --- Sub-part A: Load evaluation metrics from CSV ---
        list_of_files = glob.glob(os.path.join(experiments_dir, 'final_evaluation_results_*.csv'))
        if not list_of_files:
            raise FileNotFoundError("No 'final_evaluation_results' CSV found. Please run the 'evaluate' phase before this one.")
        
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading metrics from: {os.path.basename(latest_file)}")
        eval_df = pd.read_csv(latest_file)

        # --- Sub-part B: Calculate instance and frame counts ---
        all_instances = [inst for b in task.behaviors for inst in dataset.labels.get("labels", {}).get(b, [])]
        train_subject_set = set(train_subjects + val_subjects) # Final training used both
        test_subject_set = set(test_subjects)
        train_insts = [inst for inst in all_instances if os.path.dirname(inst['video']) in train_subject_set]
        test_insts = [inst for inst in all_instances if os.path.dirname(inst['video']) in test_subject_set]

        train_instance_counts = defaultdict(int)
        test_instance_counts = defaultdict(int)
        train_frame_counts = defaultdict(int)
        test_frame_counts = defaultdict(int)
        for inst in train_insts:
            train_instance_counts[inst['label']] += 1
            train_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
        for inst in test_insts:
            test_instance_counts[inst['label']] += 1
            test_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)

        # --- Sub-part C: Write everything to the config file ---
        config_path = dataset.config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['metrics'] = {} # Start with a fresh metrics block

        for behavior in task.behaviors:
            config['metrics'][behavior] = {}
            
            # Write performance metrics from the CSV
            f1_col, precision_col, recall_col = f'{behavior}_Test_F1', f'{behavior}_Test_Precision', f'{behavior}_Test_Recall'
            if f1_col in eval_df.columns:
                config['metrics'][behavior]['F1 Score'] = round(float(eval_df[f1_col].mean()), 2)
            if precision_col in eval_df.columns:
                config['metrics'][behavior]['Precision'] = round(float(eval_df[precision_col].mean()), 2)
            if recall_col in eval_df.columns:
                config['metrics'][behavior]['Recall'] = round(float(eval_df[recall_col].mean()), 2)

            # Write instance and frame counts
            train_n_inst = train_instance_counts.get(behavior, 0)
            train_n_frame = train_frame_counts.get(behavior, 0)
            test_n_inst = test_instance_counts.get(behavior, 0)
            test_n_frame = test_frame_counts.get(behavior, 0)
            config['metrics'][behavior]["Train Inst (Frames)"] = f"{train_n_inst} ({int(train_n_frame)})"
            config['metrics'][behavior]["Test Inst (Frames)"] = f"{test_n_inst} ({int(test_n_frame)})"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        # Update in-memory object to match
        dataset.config = config
        
        print("Successfully updated config.yaml with metrics and instance counts.")
        print("The official performance of this model is the one reported from the 'evaluate' phase.")

    except Exception as e:
        print(f"\n[ERROR] Could not update config.yaml: {e}")
        traceback.print_exc()
        print("The model.pth file was created, but the GUI card could not be updated.")


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