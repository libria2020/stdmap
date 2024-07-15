import os
import pandas
import torch

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, path, split: str, transform=None):
        self.p = pandas.read_csv(os.path.join(path, split, 'p.csv'))
        self.q = pandas.read_csv(os.path.join(path, split, 'q.csv'))

        self.transform = transform

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        p = torch.tensor(self.p.iloc[idx].values[:-1], dtype=torch.float)
        q = torch.tensor(self.q.iloc[idx].values[:-1], dtype=torch.float)

        sequence = torch.cat((p.unsqueeze(0), q.unsqueeze(0)), dim=0)
        label = torch.tensor(self.p.iloc[idx].values[-1], dtype=torch.float)

        if self.transform:
            sequence = self.transform(sequence)

        return sequence, label
