import os
import sys
import yaml
from collections import Counter

# This script needs to import the cbas and gui_state modules.
# We add the backend directory to the Python path to make this possible.
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
try:
    import cbas
    import gui_state
except ImportError as e:
    print("ERROR: Could not import CBAS modules.")
    print("Please ensure you are running this script from the root directory of your CBAS project")
    print(f"and that your virtual environment is activated. Details: {e}")
    sys.exit(1)


def analyze_split(project_path: str, dataset_name: str, seed: int):
    """
    Loads a dataset, performs a train/test split with a specific seed,
    and prints a detailed report on the composition of each set.
    """
    print("\n" + "="*50)
    print(f"--- Analysis for Split with Seed: {seed} ---")
    print("="*50)

    try:
        gui_state.proj = cbas.Project(project_path)
    except cbas.InvalidProject:
        print(f"ERROR: The provided path is not a valid CBAS project: {project_path}")
        return

    if dataset_name not in gui_state.proj.datasets:
        print(f"ERROR: Dataset '{dataset_name}' not found in the project.")
        return

    train_insts, test_insts, behaviors = gui_state.proj._load_dataset_common(
        name=dataset_name,
        split=0.2,
        seed=seed
    )

    if train_insts is None or test_insts is None:
        print("Could not load or split the dataset. The labels.yaml file might be empty.")
        return

    def get_composition_report(instances):
        if not instances:
            return set(), Counter(), Counter()

        subjects = set()
        instance_counts = Counter()
        frame_counts = Counter()

        for inst in instances:
            try:

                # The unique identifier for a subject is its full relative path.
                subject_path = os.path.dirname(inst['video']).replace('\\', '/')
                subjects.add(subject_path)

                instance_counts[inst['label']] += 1
                frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
            except Exception:
                continue
        
        return subjects, instance_counts, frame_counts

    train_subjects, train_inst_counts, train_frame_counts = get_composition_report(train_insts)
    test_subjects, test_inst_counts, test_frame_counts = get_composition_report(test_insts)

    # --- Print the final report ---
    print("\n--- TRAINING SET COMPOSITION ---")
    print(f"Subjects ({len(train_subjects)}): {', '.join(sorted(list(train_subjects)))}")
    print("\n{:<25} {:>15} {:>15}".format("Behavior", "Instances", "Frames"))
    print("-" * 57)
    for behavior in behaviors:
        print("{:<25} {:>15} {:>15}".format(
            behavior,
            train_inst_counts.get(behavior, 0),
            int(train_frame_counts.get(behavior, 0))
        ))
    print("-" * 57)
    print("{:<25} {:>15} {:>15}".format(
        "TOTAL",
        sum(train_inst_counts.values()),
        int(sum(train_frame_counts.values()))
    ))

    print("\n\n--- TEST SET COMPOSITION ---")
    print(f"Subjects ({len(test_subjects)}): {', '.join(sorted(list(test_subjects)))}")
    print("\n{:<25} {:>15} {:>15}".format("Behavior", "Instances", "Frames"))
    print("-" * 57)
    for behavior in behaviors:
        print("{:<25} {:>15} {:>15}".format(
            behavior,
            test_inst_counts.get(behavior, 0),
            int(test_frame_counts.get(behavior, 0))
        ))
    print("-" * 57)
    print("{:<25} {:>15} {:>15}".format(
        "TOTAL",
        sum(test_inst_counts.values()),
        int(sum(test_frame_counts.values()))
    ))
    print("\n" + "="*50)


if __name__ == "__main__":
    print("CBAS Train/Test Split Analyzer")
    print("This script will show you the exact composition of the training and test sets for a given random seed.")
    
    project_path_to_use = input("Please enter the FULL path to your CBAS project folder and press Enter:\n> ").strip()
    
    if not os.path.isdir(project_path_to_use):
        print(f"ERROR: Path not found: '{project_path_to_use}'")
        sys.exit(1)

    dataset_name_to_use = input("Enter the name of the dataset to analyze (e.g., cbas_aug):\n> ").strip()
    
    while True:
        try:
            seed_to_use = int(input("Enter the seed number of the run to inspect (e.g., 0, 1, 2, etc.):\n> ").strip())
            analyze_split(project_path_to_use, dataset_name_to_use, seed_to_use)
        except ValueError:
            print("Invalid input. Please enter an integer for the seed.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        another = input("\nAnalyze another seed? (y/n):\n> ").strip().lower()
        if another != 'y':
            break