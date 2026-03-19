import os
import gzip
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ATACDataset(Dataset):
    def __init__(
        self,
        dense_dir,
        sparse_dir,
        file_list=None
    ):
        """
        Args:
            dense_dir (str): directory with dense (bulk) count files
            sparse_dir (str): directory with sparse versions
            file_list (list): optional subset of filenames
        """
        self.dense_dir = dense_dir
        self.sparse_dir = sparse_dir

        # match files by name
        self.files = file_list if file_list else sorted(os.listdir(dense_dir))

        # sanity check
        self.files = [
            f for f in self.files
            if os.path.exists(os.path.join(sparse_dir, f))
        ]

    def _load_tsv(self, path):
        """Handles gzipped TSVs"""
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t")
        else:
            df = pd.read_csv(path, sep="\t")
        return df

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        dense_path = os.path.join(self.dense_dir, fname)
        sparse_path = os.path.join(self.sparse_dir, fname)

        # load data
        dense_df = self._load_tsv(dense_path)
        sparse_df = self._load_tsv(sparse_path)

        # get counts
        # dense: counts in column 1
        y = dense_df[dense_df.columns[1]].values.astype(np.float32)

        # sparse: multiple columns after bin, pick a random one
        sparse_cols = sparse_df.columns[1:]
        col = np.random.choice(sparse_cols)
        x = sparse_df[col].values.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)