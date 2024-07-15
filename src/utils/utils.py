import os
import re

from torch.utils.tensorboard import SummaryWriter

from src.dataloader.dataloader import *
from src.model.models import *

"""
Directories
"""


def create_directories(root_dir, sub_dirs):
    # construct string
    directories = [os.path.join(os.getcwd(), root_dir, name) for name in sub_dirs]

    for directory in directories:
        # create directories (all processes attempt to create the directories)
        os.makedirs(directory, exist_ok=True)

    return tuple(directories)


def version_directory(directory, gpu_id):
    # list sub-folders in directory
    dirs = [os.path.join(directory, child_dir) for child_dir in os.listdir(directory)]

    if len(dirs) != 0:
        # select the last created sub-folder
        latest = max(dirs, key=os.path.getctime)  # ../output/log/v_2

        # get the number of the last version
        num = re.search("(v_)(\d+)", latest).group(2)

        # create new version number only if master
        if gpu_id == 0:
            num = int(num) + 1

        # construct string
        child_dir = os.path.join(directory, f'v_{num}')
    else:
        # construct string
        child_dir = os.path.join(directory, 'v_0')

    #  create directory (all processes attempt to create the directory)
    os.makedirs(child_dir, exist_ok=True)

    return child_dir


"""
Model
"""

models = {'MLP': MLP, 'Conv1d': Con1DModel, 'LSTM': LSTMModel, 'Transformer': TransformerModel}


def load_model(configuration):
    """
    Create Model Object
    """
    model = models[configuration.model.name](configuration)

    if configuration.environment.distributed:
        model = model.to(configuration.trainer.gpu_id)

    return model


"""
Training Objects
"""


def load_train_objs(configuration, model):
    """
    Create Optimizer Object
    """
    optimizer = getattr(torch.optim, configuration.optimizer.type)
    optimizer = optimizer(model.parameters(), **configuration.optimizer.params)

    """
    Learning Rate Scheduler
    """
    lr_scheduler = None
    if 'lr_scheduler' in configuration.keys():
        lr_scheduler = getattr(torch.optim.lr_scheduler, configuration.lr_scheduler.type)
        lr_scheduler = lr_scheduler(optimizer, **configuration.lr_scheduler.params)

    """
    Create Loss Object
    """
    criterion = getattr(torch.nn, configuration.trainer.loss)()

    return optimizer, lr_scheduler, criterion


"""
Data
"""


def prepare_dataloader(configuration):
    """
    Create Dataloaders
    """
    dataloader = SequenceDataLoader(configuration)

    train_dataloader = dataloader.get_dataloader('train')
    val_dataloader = dataloader.get_dataloader('validation')
    test_dataloader = dataloader.get_dataloader('test')

    return train_dataloader, val_dataloader, test_dataloader


def writers(directory):
    writer = SummaryWriter(directory)
    return writer
