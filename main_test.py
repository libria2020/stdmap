# tensorboard dev upload --logdir logdir
# python3 main.py 'configuration/config.yaml'


import sys
import yaml
import pprint

from pathlib import Path
from easydict import EasyDict as edict
from tqdm import tqdm

from src.trainer.trainer import Trainer
from src.utils.checkpoint import CheckpointManager
from src.utils.enviroment import DistributedContext
from src.utils.logger import Logger
from src.utils.utils import *


def main(configuration):
    with DistributedContext(configuration):
        """
        Create Directories & versions for each run
        """

        directories = [os.path.join(configuration.dataset.output_folder, name) for name in
                       ['log', 'checkpoint', 'best_model', 'writer']]

        for directory in directories:
            # create directories (all processes attempt to create the directories)
            os.makedirs(directory, exist_ok=True)

        log_dir, ckpt_dir, best_model_dir, writer_dir = tuple(directories)

        log_dir = version_directory(log_dir, configuration.trainer.gpu_id)
        writer_dir = version_directory(writer_dir, configuration.trainer.gpu_id)

        """
        Create the SummaryWriter Object
        """
        swriter = writers(writer_dir)

        """
        Create Logger Object
        """
        logger = Logger(log_dir)

        """
        If Master log configuration
        """
        if configuration.trainer.gpu_id == 0:
            logger.log_txt('configuration', pprint.pformat(configuration))

        """
        Create CheckpointManager Object
        """
        ckpt = CheckpointManager(
            ckpt_dir,
            best_model_dir,
            'ckpt',
            logger,
            monitor=configuration.checkpoint.monitor,
            save_every=configuration.checkpoint.save_every,  # save checkpoint every time save_every is multiple of step
            max_to_keep=configuration.checkpoint.max_to_keep,
            initial_value_threshold=configuration.checkpoint.initial_value_threshold,
            verbose=configuration.checkpoint.verbose,
        )

        """
        Create Model
        """
        model = load_model(configuration)

        """
        Create Train Objects
        """
        optimizer, lr_scheduler, criterion = load_train_objs(configuration, model)

        """
        Create Dataloaders
        """
        dataloader = SequenceDataLoader(configuration)
        test_dataloader = dataloader.get_dataloader('test')

        """
        Create Trainer Object
            where all the magic happens
        """
        trainer = Trainer(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            configuration.trainer.metrics,
            configuration.environment.distributed,
            configuration.trainer.gpu_id,
            ckpt,
            logger,
            swriter,
            configuration.trainer.epochs,
            configuration.trainer.save_img_every,  # save images every time save_every is multiple of step
            configuration.trainer.verbose
        )

        trainer.predict(test_dataloader)

        swriter.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [YAML CONFIGURATION FILE]")
        sys.exit(0)

    path = Path(sys.argv[1])

    # path = Path("configuration/lstm_test.yaml")

    """
    Load Configuration File
    """
    if not path.exists():
        print("Configuration file does not exist! Run again with a valid configuration file!")
        sys.exit(0)

    with path.open('r') as f:
        configuration = edict(yaml.safe_load(f))

    root = os.path.join(configuration.dataset.output_folder, str(configuration.dataset.sequence_length))

    for trajectories in tqdm([1 if i == 0 else i * 10 for i in range(17)]):
        print(trajectories)
        configuration.dataset.num_trajectories = trajectories
        configuration.dataset.output_folder = root + '_' + str(trajectories)

        main(configuration)

    # for sequence in tqdm([128, 256, 512, 1024, 2048]):
    #     print(sequence)
    #     configuration.dataset.sequence_length = sequence
    #     configuration.dataset.output_folder = root + '_' + str(sequence)
    #
    #     main(configuration)

    print("Finished!")
