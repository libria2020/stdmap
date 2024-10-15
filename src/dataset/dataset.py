import os
import pandas
import torch

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, path, split: str, transform=None, trajectories='all'):
        p = pandas.read_csv(os.path.join(path, split, 'p.csv'))
        q = pandas.read_csv(os.path.join(path, split, 'q.csv'))

        if split == 'test':
            if trajectories == 'all':
                self.p = p
                self.q = q
            elif trajectories == 'chaotic':
                self.p = p[p['chaotic'] == 1]
                self.q = q[q['chaotic'] == 1]
            elif trajectories == 'non_chaotic':
                self.p = p[p['chaotic'] == 0]
                self.q = q[q['chaotic'] == 0]

            if 'lyapunov_exponent' in self.p.columns:
                self.p = self.p.drop(columns=['lyapunov_exponent'])
                self.q = self.q.drop(columns=['lyapunov_exponent'])
            if 'chaotic' in self.p.columns:
                self.p = self.p.drop(columns=['chaotic'])
                self.q = self.q.drop(columns=['chaotic'])
        else:
            self.p = p
            self.q = q

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
