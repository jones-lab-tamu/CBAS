"""
Defines the data splitting abstraction for CBAS.

This module provides a consistent way to partition datasets at the subject level,
ensuring splits are group-aware, stratified, and validated. This prevents
code duplication and allows both the GUI and headless scripts to use the same
robust data handling pipeline.
"""

import numpy as np
import random
import os
import json
import hashlib
import time
from collections import defaultdict

def _generate_dataset_fingerprint(dataset):
    """
    Creates a unique signature for a dataset based on its subjects and label counts.
    This ensures that a set of splits is only ever used with the exact dataset
    it was created for.
    """
    subjects = set()
    label_counts = defaultdict(int)
    all_instances = [inst for b_labels in dataset.labels.get("labels", {}).values() for inst in b_labels]

    for inst in all_instances:
        # Use the directory name as the unique subject identifier
        subjects.add(os.path.dirname(inst['video']))
        label_counts[inst['label']] += 1

    # Create a stable, sorted representation to ensure consistent hashing
    sorted_subjects = sorted(list(subjects))
    sorted_labels = sorted(label_counts.items())

    # Hash the string representation of the sorted lists
    hasher = hashlib.md5()
    hasher.update(str(sorted_subjects).encode('utf-8'))
    hasher.update(str(sorted_labels).encode('utf-8'))
    return hasher.hexdigest()


class SplitProvider:
    """Abstract base class for data splitters."""
    def get_split(self, run_index: int, all_subjects: list, all_instances: list, behaviors: list) -> tuple[list, list, list]:
        """Returns lists of subject IDs for train, validation, and test sets for a given run index."""
        raise NotImplementedError

class RandomSplitProvider(SplitProvider):
    """
    Generates group-aware, stratified splits on the fly using a random seed.
    For each run_index, it generates a NEW random split.
    """
    def __init__(self, seed=None, split_ratios=(0.70, 0.15, 0.15), stratify=True):
        self.initial_seed = seed if seed is not None else int(time.time())
        self.ratios = split_ratios
        self.stratify = stratify

    def _is_split_valid(self, train_insts, val_insts, all_behaviors):
        """Checks if train and val sets contain all behaviors."""
        train_behaviors = {inst['label'] for inst in train_insts}
        val_behaviors = {inst['label'] for inst in val_insts}
        return train_behaviors == all_behaviors and val_behaviors == all_behaviors

    def get_split(self, run_index: int, all_subjects: list, all_instances: list, behaviors: list) -> tuple[list, list, list]:
        # Use the run_index to ensure each run gets a different, reproducible seed.
        current_seed = self.initial_seed + run_index
        rng = np.random.default_rng(current_seed)
        
        # It shuffles, splits, and validates.
        subject_to_insts = defaultdict(list)
        for inst in all_instances:
            subject = os.path.dirname(inst['video'])
            subject_to_insts[subject].append(inst)

        for attempt in range(10):
            shuffled_subjects = list(all_subjects)
            rng.shuffle(shuffled_subjects)
            n_total = len(shuffled_subjects)
            n_train = int(self.ratios[0] * n_total)
            n_val = int(self.ratios[1] * n_total)
            train_subjects = shuffled_subjects[:n_train]
            val_subjects = shuffled_subjects[n_train : n_train + n_val]
            test_subjects = shuffled_subjects[n_train + n_val:]
            if self.ratios[2] == 0.0 and (n_train + n_val) < n_total:
                val_subjects = shuffled_subjects[n_train:]
            if self.stratify:
                train_insts = [inst for s in train_subjects for inst in subject_to_insts[s]]
                val_insts = [inst for s in val_subjects for inst in subject_to_insts[s]]
                if self._is_split_valid(train_insts, val_insts, set(behaviors)):
                    return train_subjects, val_subjects, test_subjects
            else:
                return train_subjects, val_subjects, test_subjects
            rng = np.random.default_rng(current_seed + attempt + 1)
        
        raise RuntimeError("Failed to generate a valid stratified split after 10 attempts.")


class ManifestSplitProvider(SplitProvider):
    """
    Provides splits by reading them from a pre-computed manifest file.
    The run_index corresponds to the index in the manifest's 'splits' list.
    """
    def __init__(self, manifest_path: str, dataset_fingerprint: str):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        if self.manifest.get('dataset_fingerprint') != dataset_fingerprint:
            raise ValueError("FATAL: Dataset fingerprint in manifest does not match current dataset. The splits are not valid for this data.")

    def get_split(self, run_index: int, all_subjects: list, all_instances: list, behaviors: list) -> tuple[list, list, list]:
        """
        Retrieves a specific split from the manifest using the provided run_index.
        The other arguments are ignored but included for signature compatibility.
        """
        if not 0 <= run_index < len(self.manifest['splits']):
            raise IndexError(f"Run index {run_index} is out of bounds for manifest with {len(self.manifest['splits'])} splits.")
        
        replicate_data = self.manifest['splits'][run_index]
        train_subjects = replicate_data['train']
        val_subjects = replicate_data['validation']
        test_subjects = replicate_data['test']
        return train_subjects, val_subjects, test_subjects