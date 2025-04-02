import os
import csv
import h5py
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

def load_pdac_data(dataset_root, cv_fold):
    """Load PDAC dataset for cross-validation.

    Args:
        dataset_root (str): Path to the dataset root directory.
        cv_fold (int): Number of cross-validation folds.

    Returns:
        tuple: Returns lists for training, testing, and validation data.
    """
    fold_root = os.path.join(dataset_root, f'fold{cv_fold}')
    fold_files = [os.path.join(fold_root, f'fold_{i}.csv') for i in range(1, cv_fold + 1)]
    folds = []

    # Loop over each fold and get patient labels
    for fold_file in fold_files:
        p, l = get_patient_label(fold_file)
        folds.append((p, l))

    train_p, train_l, test_p, test_l = [], [], [], []

    # Split data into train and test for each fold
    for fold_idx in range(cv_fold):
        test_p_fold, test_l_fold = folds[fold_idx]
        train_p_fold, train_l_fold = [], []

        # Collect data for training from other folds
        for i, (p, l) in enumerate(folds):
            if i != fold_idx:
                train_p_fold.extend(p)
                train_l_fold.extend(l)

        train_p.append(np.array(train_p_fold, dtype=object))
        train_l.append(np.array(train_l_fold, dtype=object))
        test_p.append(np.array(test_p_fold, dtype=object))
        test_l.append(np.array(test_l_fold, dtype=object))
        val_p, val_l = [], []  # Placeholder for validation data

    return train_p, train_l, test_p, test_l, val_p, val_l

def readCSV(filename):
    """Read a CSV file and return its contents as a list of lines.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        list: List of lines from the CSV file.
    """
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    """Extract patient and label information from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing patient data.

    Returns:
        tuple: Arrays of patient identifiers and corresponding labels.
    """
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)

    # Extract patient IDs and labels from CSV data
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    
    # Print the count of labels
    a = Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))

    return np.array(patients_list, dtype=object), np.array(labels_list, dtype=object)


class PDACDataset(Dataset):
    """PDAC dataset class for loading features and labels.

    Args:
        file_name (list): List of filenames for the dataset.
        file_label (list): List of corresponding labels.
        root (str): Root directory where the features are stored.
        persistence (bool): Flag to enable feature persistence (caching).
        keep_same_psize (int): Option to keep patches of the same size.
        is_train (bool): Flag indicating whether the dataset is for training.
    """
    def __init__(self, file_name, file_label, root, persistence=False, keep_same_psize=0, is_train=False):
        super(PDACDataset, self).__init__()
        self.file_name = file_name
        self.slide_label = [int(_l) for _l in file_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # If persistence is enabled, load the features into memory
        if persistence:
            self.feats = []
            for fname in self.file_name:
                file_path = os.path.join(root, "feats_ImageNet_norm", f"{fname}.h5")
                with h5py.File(file_path, "r") as h5_file:
                    self.feats.append(torch.tensor(h5_file["features"][:]))

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """Retrieve the features and label for a specific index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing features, label, and slide ID.
        """
        if self.persistence:
            # Retrieve the features from cached memory if persistence is enabled
            features = self.feats[idx]
        else:
            # Load features from the corresponding H5 file
            file_path = os.path.join(self.root, "feats_ImageNet_norm", f"{self.file_name[idx]}.h5")
            with h5py.File(file_path, "r") as h5_file:
                features = torch.tensor(h5_file["features"][:])
        
        label = int(self.slide_label[idx])  # Convert label to integer

        slide_id = self.file_name[idx]  # Retrieve slide ID

        return features, label, slide_id  # Return the features, label, and slide ID
