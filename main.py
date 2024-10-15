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
        log_dir, ckpt_dir, best_model_dir, writer_dir = create_directories(
            configuration.dataset.output_folder, ['log', 'checkpoint', 'best_model', 'writer'])

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
        train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(configuration)

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

        trainer.fit(train_dataloader, val_dataloader)

        trainer.predict(test_dataloader)

        swriter.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [YAML CONFIGURATION FILE]")
        sys.exit(0)

    path = Path(sys.argv[1])

    # path = Path("configuration/cnn.yaml")

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


    # --- Different Loss Function ---
    # with path.open('r') as f:
    #     configuration = edict(yaml.safe_load(f))
    # root = os.path.join(
    #     configuration.dataset.output_folder,
    #     configuration.trainer.loss,
    #     str(configuration.dataset.sequence_length)
    # )
    #
    # trajectories = [1] + numpy.arange(10, 170, 10).tolist()
    # for trajectory in tqdm(trajectories):
    #
    #     configuration.dataset.num_trajectories = trajectory
    #     configuration.dataset.output_folder = root + '_' + str(trajectory)
    #
    #     main(configuration)


    print("Finished!")


# TODO finish lstm model and add transformer model
# TODO: prepared a code that computes in parallel the exponent for all trajectories
# TODO: redo all figures of the paper with other values of k
# TODO: lyapunov exponents are computed for all trajectories of test set ks add the labels to the datatset
# TODO write comment for 2048_sequence v1 test 02 v2 test 03 v3 test 04
# TODO do test with cnn and with separate orbits


# def compute_threshold(lyapunov_exponents: np.ndarray) -> float:
#     """
#     Computes the threshold for chaotic trajectories based on Lyapunov exponents.
#
#     Parameters:
#     - lyapunov_exponents: np.ndarray, Lyapunov exponents for the trajectories.
#
#     Returns:
#     - threshold: float, the computed threshold value.
#     """
#     # Number of bins using Freedman-Diaconis rule
#     N = len(lyapunov_exponents)
#     Q3 = np.quantile(lyapunov_exponents, 0.75)
#     Q1 = np.quantile(lyapunov_exponents, 0.25)
#     IQR = Q3 - Q1
#     h = 2 * IQR / pow(N, 1 / 3)  # Bin width
#     bins = (np.max(lyapunov_exponents) - np.min(lyapunov_exponents)) / h
#     bins = math.ceil(bins)
#
#     # Create histogram
#     counts, bin_edges = np.histogram(lyapunov_exponents, bins=bins)
#
#     # Find peaks in the histogram
#     peaks, _ = find_peaks(counts)
#
#     # Ensure there are at least two peaks
#     if len(peaks) < 2:
#         raise ValueError("Less than two peaks found in the histogram.")
#
#     # Get the x-values of the peaks (bin centers corresponding to peaks)
#     values = (bin_edges[peaks] + bin_edges[peaks + 1]) / 2
#
#     # Calculate the threshold as the mean between the smallest and highest peaks on the x-axis
#     threshold = (np.min(lyapunov_exponents) + np.max(lyapunov_exponents)) / 2
#     return threshold
#
#
# def plot_phase_space(p: np.ndarray, q: np.ndarray, chaotic_trajectories: np.ndarray,
#                      non_chaotic_trajectories: np.ndarray):
#     """
#     Plots the phase space (magnetogram) for chaotic and non-chaotic trajectories.
#
#     Parameters:
#     - p: np.ndarray, momentum values.
#     - q: np.ndarray, position values.
#     - chaotic_trajectories: np.ndarray, boolean mask for chaotic trajectories.
#     - non_chaotic_trajectories: np.ndarray, boolean mask for non-chaotic trajectories.
#     """
#     plt.figure(figsize=(8, 6))
#     plt.scatter(q[:, non_chaotic_trajectories], p[:, non_chaotic_trajectories], s=1, color='r', alpha=0.15)
#     plt.scatter(q[:, chaotic_trajectories], p[:, chaotic_trajectories], s=1, color='b', alpha=0.15)
#     plt.title("Chaotic and Non-Chaotic Trajectories (Phase Space)")
#     plt.xlabel("q")
#     plt.ylabel("p")
#     plt.show()
#
#
# def plot_histogram_with_threshold(lyapunov_exponents: np.ndarray, threshold: float, idx: int):
#     """
#     Plots the histogram of Lyapunov exponents and adds a vertical line at the threshold.
#
#     Parameters:
#     - lyapunov_exponents: np.ndarray, Lyapunov exponents for the trajectories.
#     - threshold: float, the threshold value.
#     - idx: int, index for saving the plot.
#     """
#     # Number of bins using Freedman-Diaconis rule
#     N = len(lyapunov_exponents)
#     Q3 = np.quantile(lyapunov_exponents, 0.75)
#     Q1 = np.quantile(lyapunov_exponents, 0.25)
#     IQR = Q3 - Q1
#     h = 2 * IQR / pow(N, 1 / 3)  # Bin width
#     bins = (np.max(lyapunov_exponents) - np.min(lyapunov_exponents)) / h
#     bins = math.ceil(bins)
#     bins = 100 if bins > 1500 else bins
#
#     # Plot the histogram of Lyapunov exponents
#     plt.figure(figsize=(6, 6))
#     plt.hist(lyapunov_exponents, bins=bins, color='blue', edgecolor='black', alpha=0.7)
#
#     # Adding labels and title
#     plt.title('Histogram of Lyapunov Exponents', fontsize=16)
#     plt.xlabel('Lyapunov Exponent', fontsize=14)
#     plt.ylabel('Frequency', fontsize=14)
#
#     # Draw a vertical line at the threshold
#     plt.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
#
#     # Display the legend and grid
#     plt.legend()
#     plt.grid(True)
#
#     # Save the plot in the "histograms" directory
#     plt.savefig(os.path.join('histograms', f'{idx}.png'), dpi=300, bbox_inches='tight')

