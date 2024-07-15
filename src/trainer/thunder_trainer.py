"""
Training with PyTorch
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

Distributed Data Parallel in PyTorch Tutorial Series
https://www.youtube.com/playlist?list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj
"""
import torch

import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from enum import Enum

from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Logger
from src.utils.meter import Metrics


class Mode(Enum):
    TRAIN = 'train'
    VALIDATE = 'validation'
    TEST = 'test'


class ThunderTrainer(object):
    def __init__(
            self,
            model: nn.Module,
            criterion,
            optimizer,
            lr_scheduler,
            metric: str,
            distributed: bool,
            gpu_id: int,
            ckpt: CheckpointManager,
            logger: Logger,
            writer: SummaryWriter,
            epochs: int = 100,
            save_img_every: int = 10,
            verbose: bool = True
    ):
        self.distributed = distributed
        self.gpu_id = gpu_id
        self.is_master = self.gpu_id == 0

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric = metric
        self.epochs = epochs

        self.last_epoch = 0
        self.current_epoch = 0
        self.global_step = 0

        self.metric_tracker = {mod.value: Metrics(self.metric) for mod in Mode}

        self.logger = logger
        self.writer = writer
        self.ckpt = ckpt

        self.save_img_every = save_img_every
        self.verbose = verbose

        # load checkpoint
        self._load_checkpoint()

        if self.distributed:
            # before training, we need to wrapped our model with dpp
            # gpu_id is the GPU in which our models lives on
            self.model = DDP(self.model, device_ids=[self.gpu_id])

        # log model architecture & latent space dimension
        if self.is_master:
            logger.log_txt('model', model.__str__())

    def _forward(self, x):
        raise NotImplementedError

    def _training_step(self, batch, batch_index):
        raise NotImplementedError

    def _validation_step(self, batch, batch_index):
        raise NotImplementedError

    def _test_step(self, batch, batch_index):
        raise NotImplementedError

    def _run_epoch(self, dataloader: DataLoader, training_state):
        # set correct metric tracker & reset values
        metric = self.metric_tracker[training_state]
        metric.reset()

        if self.distributed:
            dataloader.sampler.set_epoch(self.current_epoch)

        for batch_index, batch in enumerate(dataloader):
            self.global_step = batch_index + self.current_epoch * len(dataloader)

            # TRAIN one batch
            if training_state == Mode.TRAIN.value:
                # forward and backward pass over the train dataset
                train_loss = self._training_step(batch, batch_index)
                # update metric tracker
                score = train_loss.detach().cpu().item()
                metric += score

                # [-----------------------------------------------------------------------------------------------------
                # if master print one step & and save model if metric has improved
                if self.is_master and self.verbose == True:
                    self._print_step(training_state, score, batch_index, len(dataloader))
                # -----------------------------------------------------------------------------------------------------]

            # VALIDATE one batch
            elif training_state == Mode.VALIDATE.value:
                # disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    val_loss = self._validation_step(batch, batch_index)
                    score = val_loss.detach().cpu().item()
                    metric += score

                # [-----------------------------------------------------------------------------------------------------
                if self.is_master and self.verbose == True:
                    self._print_step(training_state, score, batch_index, len(dataloader))
                # -----------------------------------------------------------------------------------------------------]

            # TEST one batch
            elif training_state == Mode.TEST.value:
                # disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    test_loss = self._test_step(batch, batch_index)
                    score = test_loss.detach().cpu().item()
                    metric += score

                # [=====================================================================================================
                if self.is_master and self.verbose == True:
                    print(
                        f'[{training_state.upper()}][STEP: {batch_index + 1 :5d}/{len(dataloader)}] '
                        f'{self.metric.upper()}: {score :.8f}', end="\n"
                    )
                # =====================================================================================================]

        return metric

    def fit(self, train_data: DataLoader, validation_data: DataLoader):
        for epoch in range(self.last_epoch, self.epochs):
            self.current_epoch = epoch

            # TRAIN one epoch
            # set gradient tracking on
            self.model.train()
            train_metric = self._run_epoch(train_data, Mode.TRAIN.value)

            # VALIDATE one epoch
            # set the model to evaluation mode, disabling dropout
            # and using population statistics for batch normalization.
            self.model.eval()
            val_metric = self._run_epoch(validation_data, Mode.VALIDATE.value)

            # [---------------------------------------------------------------------------------------------------------
            # if master print & log epoch
            if self.is_master and self.verbose:
                self._print_epoch()
                self._log_epoch()
            # ---------------------------------------------------------------------------------------------------------]

            value = val_metric.score

            # LEARNING RATE step
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(value)
                else:
                    if self.optimizer.param_groups[0]['lr'] >= 1.1e-5:
                        self.lr_scheduler.step()

            # [---------------------------------------------------------------------------------------------------------
            # if master save checkpoint
            if self.is_master:
                self.ckpt.save(self._get_checkpoint())
                self.ckpt.save_best(self._get_checkpoint(), self.current_epoch, value)
            # ---------------------------------------------------------------------------------------------------------]

    def predict(self, test_data: DataLoader):
        # TEST model
        # set the model to evaluation mode, disabling dropout
        # and using population statistics for batch normalization.
        self.model.eval()
        test_metric = self._run_epoch(test_data, Mode.TEST.value)

        # [=============================================================================================================
        # if master print, log
        if self.is_master:
            if self.verbose:
                print(f'[TEST]  {test_metric.name.upper()}: {test_metric.score:.8f}')

            self.logger.log_csv(Mode.TEST.value, 'test_loss', test_metric.score, self.current_epoch)
            self.writer.add_scalar('test loss', test_metric.score, self.current_epoch)
        # =============================================================================================================]

    def _get_checkpoint(self):
        checkpoint = {
            "model_sate": self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "current_epoch": self.current_epoch,
        }

        return checkpoint

    def _load_checkpoint(self):
        loc = f"cuda:{self.gpu_id}" if self.distributed else None
        checkpoint = self.ckpt.load(loc)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_sate"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"]) if checkpoint[
                                                                                 "lr_scheduler"] is not None else None
            self.last_epoch = checkpoint["current_epoch"] + 1

    def _log_epoch(self):
        train_metric = self.metric_tracker[Mode.TRAIN.value]
        val_metric = self.metric_tracker[Mode.VALIDATE.value]

        self.logger.log_csv(Mode.TRAIN.value, 'train_loss', train_metric.score, self.current_epoch)
        self.logger.log_csv(Mode.VALIDATE.value, 'val_loss', val_metric.score, self.current_epoch)

        self.writer.add_scalars('loss',
                                {'training loss': train_metric.score, 'validation loss': val_metric.score},
                                self.current_epoch)

        if self.lr_scheduler:
            self.logger.log_csv('learning_rate',
                                'learning_rate',
                                self.optimizer.param_groups[0]['lr'],
                                self.current_epoch)

            self.writer.add_scalar('learning_rate',
                                   self.optimizer.param_groups[0]['lr'],
                                   self.current_epoch)

    def _print_epoch(self):
        train_metric = self.metric_tracker[Mode.TRAIN.value]
        val_metric = self.metric_tracker[Mode.VALIDATE.value]

        print(
            f'[EPOCH: {self.current_epoch + 1 :3d}/{self.epochs}] [{"train".upper()}] '
            f'{train_metric.name.upper()}: {train_metric.score:.8f} | '
            f'[{"val".upper()}] '
            f'{val_metric.name.upper()}: {val_metric.score:.8f} | '
            f'lr: {self.optimizer.param_groups[0]["lr"]}', end="\n"
        )

    def _print_step(self, training_state, score, part, total):
        print(
            f'[{training_state.upper()}][EPOCH: {self.current_epoch + 1 :3d}/{self.epochs}]'
            f'[STEP: {part + 1 :5d}/{total}] '
            f'{self.metric.upper()}: {score :.8f} | '
            f'lr: {self.optimizer.param_groups[0]["lr"]}', end="\n"
        )
