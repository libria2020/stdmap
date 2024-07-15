import glob
import os
import re
import numpy
import torch

from src.utils.logger import Logger


class CheckpointManager:
    def __init__(self,
                 checkpoint_dir: str,  # .../checkpoint
                 best_model_dir: str,  # .../best_model
                 filename: str,
                 logger: Logger,
                 monitor: str = 'loss',
                 save_every: int = 1,
                 max_to_keep: int = 3,
                 initial_value_threshold=None,
                 verbose: int = 1
                 ):
        """
        Saved a checkpoint at the end of each epoch in the `ckpt_dir`, replacing the existing checkpoint
        This checkpoint is used to resume training

        Each time "best model" is saved a line in the `log/best_model.txt` file is record by the logger
        to keep track of the monitored metric

        Also keeps track of the last best models
            monitor: 	             The metric name to monitor.
                                     When metric improves the model is saved in the `best_model_dir`
            save_every:              Saves best model at end of this many epochs
            max_to_keep:             How many best model checkpoints keep
            initial_value_threshold: Initial "best" value of the metric to be monitored
            verbose:                 Verbosity mode, 0 or 1.
                                     Mode 0 is silent, and mode 1 display messages when the models are saved
        """

        self.monitor = monitor
        self.value = numpy.inf

        self.save_every = save_every
        self.max_to_keep = max_to_keep
        self.initial_value_threshold = initial_value_threshold
        self.verbose = verbose
        self.logger = logger

        self.filename = filename
        self.checkpoint_path = os.path.join(checkpoint_dir, f'{self.filename}.pt')  # .../checkpoint/ckpt.pt

        self.best_model_dir = best_model_dir  # .../best_model

        self.best_model_previous = None
        self.best_model_current = -1

        self._best_model()

    def _best_model(self):
        directories = glob.glob(os.path.join(self.best_model_dir, 'ckpt*'))
        if len(directories) != 0:
            latest = max(directories, key=os.path.getctime)  # .../best_model/ckpt_1.pt
            pattern = f'({self.filename}_)(\d+)'
            num = re.search(pattern, latest).group(2)
            self.best_model_current = int(num)  # self.best_model_current = 1
            # save best metric value
            with open(os.path.join(self.best_model_dir, 'value.npy'), 'rb') as f:
                self.value = numpy.load(f)[0]

    def save_best(self, checkpoint, epoch, value: float):
        if epoch % self.save_every == 0:
            if value < self.initial_value_threshold and value < self.value:
                self.value = value

                # self.best_model_previous = 1
                self.best_model_previous = self.best_model_current
                # self.best_model_current = 2
                self.best_model_current = (self.best_model_current + 1) % self.max_to_keep

                # .../best_model/ckpt_2
                path = os.path.join(self.best_model_dir, f'{self.filename}_{self.best_model_current}')
                # save model
                torch.save(checkpoint, path)
                # save best metric value
                with open(os.path.join(self.best_model_dir, 'value.npy'), 'wb') as f:
                    numpy.save(f, numpy.array([value]))

                msg = f'Epoch {checkpoint["current_epoch"]} | Best Model saved at `{path}` | ' \
                      f'Monitored Metric: {self.monitor} - {value :.8f}'

                self.logger.log_txt("best_model", msg)

                if self.verbose:
                    print(msg)

    def save(self, checkpoint):
        torch.save(checkpoint, self.checkpoint_path)  # .../checkpoint/ckpt.pt

        if self.verbose:
            print(f'Epoch {checkpoint["current_epoch"]} | Training checkpoint saved at `{self.checkpoint_path}`')

    def load(self, loc):
        checkpoint = None

        if os.path.exists(self.checkpoint_path):  # .../checkpoint/ckpt.pt
            checkpoint = torch.load(self.checkpoint_path, map_location=loc)

            if self.verbose:
                print(
                    f'Resuming training from checkpoint `{self.checkpoint_path}` '
                    f'last saved at Epoch {checkpoint["current_epoch"]}'
                )

        else:
            if self.verbose:
                print(f'No checkpoint present at `{self.checkpoint_path}`. Training from scratch!')

        return checkpoint