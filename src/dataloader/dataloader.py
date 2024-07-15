import os
import pandas
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import transforms

from src.dataset.dataset import SequenceDataset


class SequenceNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x[0, :] = (x[0, :] - self.mean[0]) / self.std[0]
        x[1, :] = (x[1, :] - self.mean[1]) / self.std[1]
        return x


class SequenceDataLoader:
    def __init__(self, configuration):
        self.dataloader_params = configuration.dataloader

        self.distributed = configuration.environment.distributed

        self.gpu_id = configuration.trainer.gpu_id
        self.is_master = self.gpu_id == 0

        self.dataset = os.path.join(
            configuration.dataset.input_folder,
            str(configuration.dataset.sequence_length),
            str(configuration.dataset.num_trajectories)
        )

        self.mean_std = pandas.read_csv(
            os.path.join(self.dataset, 'train', 'mean_std.csv'),
            index_col=0
        )

    def _transformations(self):
        p_mean = self.mean_std.loc['p', 'mean']
        p_std = self.mean_std.loc['p', 'std']

        q_mean = self.mean_std.loc['q', 'mean']
        q_std = self.mean_std.loc['q', 'std']

        transformations = transforms.Compose([SequenceNormalize([p_mean, q_mean], [p_std, q_std])])

        return transformations

    def _get_dataset(self, text):
        if text == 'train':
            return SequenceDataset(self.dataset, text, transform=self._transformations())
        elif text == 'validation':
            return SequenceDataset(self.dataset, text, transform=self._transformations())
        elif text == 'test':
            return SequenceDataset(self.dataset, text, transform=self._transformations())

    def get_dataloader(self, text):
        dataset = self._get_dataset(text)

        sampler = None
        shuffle = True

        if self.distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.dataloader_params.batch_size,
            pin_memory=True,
            shuffle=shuffle,
            num_workers=self.dataloader_params.num_workers,
            sampler=sampler
        )
