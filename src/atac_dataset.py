import os
import gzip
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ATACDataset(Dataset):
    def __init__(
        self,
        atac_dir,
        sparsity,
        sparse_function=None,
        file_list=None
    ):
        """
        Args:
            dense_dir (str): directory with dense (bulk) count files
            sparse_dir (str): directory with sparse versions
            file_list (list): optional subset of filenames
        """
        self.atac_dir = atac_dir
        self.sparsity = sparsity
        self.sparse_function = sparse_function

        # match files by name
        self.files = file_list if file_list else sorted(os.listdir(atac_dir))

    def _load_tsv(self, path):
        """Handles gzipped TSVs"""
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t")
        else:
            df = pd.read_csv(path, sep="\t")
        cols = [i for i in df.columns if f"_{self.sparsity}_" not in i]
        return df[cols]
    
    def _load_sparse_tsv(self, path):
        """Handles gzipped TSVs"""
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t")
        else:
            df = pd.read_csv(path, sep="\t")
        cols = [i for i in df.columns if f"_{self.sparsity}_" in i]
        return df[cols]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        atac_path = os.path.join(self.atac_dir, fname)

        # load data
        dense_df = self._load_tsv(atac_path)

        # get counts
        # dense: counts in column 1
        y = dense_df[dense_df.columns[1]].values.astype(np.float32)

        # sparse: depends on sparsity function
        # if sparsing function provided, use that
        if self.sparse_function is not None:
            x = self.sparse_function(y, self.sparsity)
         # if none, pick a random sparse colum
        else:
            sparse_df = self._load_sparse_tsv(atac_path)
            sparse_cols = sparse_df.columns[:]
            col = np.random.choice(sparse_cols)
            x = sparse_df[col].values.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)

class SampleReads:
    
    def __call__(self, x, sparsity):

        sparse_counts = np.random.binomial(
            x.astype(int),
            sparsity
        )

        return sparse_counts.astype(np.float32)

def create_dataloader(
    atac_dir,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    sparsity=1,
    val_split=0.2,
    seed=42
):
    all_files = sorted(os.listdir(atac_dir))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)

    split_idx = int(len(all_files) * (1 - val_split))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    train_dataset = ATACDataset(
        atac_dir=atac_dir,
        sparsity=sparsity,
        sparse_function=None,#sparse_function=SampleReads(),
        file_list=train_files
    )

    val_dataset = ATACDataset(
        atac_dir=atac_dir,
        sparsity=sparsity,
        sparse_function=None,#sparse_function=SampleReads(),
        file_list=val_files
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # IMPORTANT
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader