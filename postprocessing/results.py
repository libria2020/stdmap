import sys
import pandas
import yaml
import pprint
import os
import numpy

from pathlib import Path
from easydict import EasyDict as edict
from tqdm import tqdm
import seaborn as sns

from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt


def plot_learning_curves(configuration, root):
    # for trajectories in tqdm(tqdm([1 if i == 0 else i * 10 for i in range(17)])):
    for sequence in tqdm([128, 256, 512, 1024, 2048]):
        # configuration.dataset.num_trajectories = trajectories
        # configuration.dataset.output_folder = root + '_' + str(trajectories)

        os.makedirs(os.path.join(root, 'learning_curves'), exist_ok=True)
        filename = os.path.join(root, 'learning_curves', f'{sequence}')

        configuration.dataset.sequence_length = sequence
        configuration.dataset.output_folder = root + '_' + str(sequence)

        path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

        train_file = os.path.join(path, 'train.csv')
        validation_file = os.path.join(path, 'validation.csv')

        df_train_loss = pandas.read_csv(train_file)
        df_validation_loss = pandas.read_csv(validation_file)

        epochs = df_train_loss.epoch.values
        train_loss = df_train_loss.value.values
        validation_loss = df_validation_loss.value.values

        train_loss_Spline = make_interp_spline(epochs, train_loss)
        validation_loss_Spline = make_interp_spline(epochs, validation_loss)

        epochs_ = numpy.linspace(epochs.min(), epochs.max(), 500)
        train_loss_ = train_loss_Spline(epochs_)
        validation_loss_ = validation_loss_Spline(epochs_)

        fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))
        ax.plot(epochs_, train_loss_, label='train loss')
        ax.plot(epochs_, validation_loss_, label='validation loss')

        ax.set_xlabel('epochs', fontsize=12)
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)

        fig.suptitle(f'Sequence Length (L) = {configuration.dataset.sequence_length}, Number of Sequences (N)'
                     f' = {configuration.dataset.num_trajectories}', fontsize=14)

        plt.legend(loc='best', fontsize=12)
        # plt.show()
        plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.close()


def plot_test_mse(configuration, root):
    number_of_sequences = []
    mse = []

    os.makedirs(os.path.join(root, 'test_set_error'), exist_ok=True)
    filename = os.path.join(root, 'test_set_error', 'test_set_error')

    # for trajectories in tqdm(tqdm([1 if i == 0 else i * 10 for i in range(17)])):
    for sequence in tqdm([128, 256, 512, 1024, 2048]):
        # configuration.dataset.num_trajectories = trajectories
        # configuration.dataset.output_folder = root + '_' + str(trajectories)

        configuration.dataset.sequence_length = sequence
        configuration.dataset.output_folder = root + '_' + str(sequence)

        path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

        test_file = os.path.join(path, 'test.csv')
        df_test_loss = pandas.read_csv(test_file)

        # number_of_sequences.append(trajectories)
        number_of_sequences.append(sequence)
        mse.append(df_test_loss.value.values[0])

    fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))

    # ax.plot(number_of_sequences[1:], mse[1:], "ok", label='test loss')
    ax.plot(number_of_sequences, mse, "ok", label='test loss')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    # number_of_sequences_Spline = make_interp_spline(number_of_sequences[1:], mse[1:])
    number_of_sequences_Spline = make_interp_spline(number_of_sequences, mse)


    # number_of_sequences_ = numpy.linspace(numpy.min(number_of_sequences[1:]), numpy.max(number_of_sequences[1:]), 500)
    number_of_sequences_ = numpy.linspace(numpy.min(number_of_sequences), numpy.max(number_of_sequences), 500)
    mse_ = number_of_sequences_Spline(number_of_sequences_)
    # ax.plot(number_of_sequences_, mse_, label='test loss')

    # ax.set_xlabel('Number of Sequences (N)', fontsize=12)
    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)

    # fig.suptitle(f'Sequence Length (L) = {configuration.dataset.sequence_length}', fontsize=14)
    fig.suptitle(f'Number of Trajectories (N) = {configuration.dataset.num_trajectories}', fontsize=14)

    plt.axhline(y=0.05, color='r', linestyle='--')

    plt.legend(loc='best', fontsize=12)
    # plt.show()
    plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(configurtion, root):
    for trajectories in tqdm(tqdm([1 if i == 0 else i * 10 for i in range(17)])):
        configuration.dataset.num_trajectories = trajectories
        configuration.dataset.output_folder = root + '_' + str(trajectories)

        path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

        predictions_file = os.path.join(path, 'predictions.csv')
        df_predictions = pandas.read_csv(predictions_file)

        reals = df_predictions.real.values
        predictions = df_predictions.predictions.values

        k_reals = []
        k_predictions = []
        k_error = []

        for k in numpy.unique(reals):
            reals_k = reals[reals == k]
            predictions_k = predictions[reals == k]

            mean = numpy.mean(predictions_k)
            std = numpy.std(predictions_k)

            k_reals.append(k)
            k_predictions.append(mean)
            k_error.append(std)

        fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))

        # line = [i for i in range(0, 11, 1)]

        line = numpy.unique(reals)

        ax.plot(line, line, "k-", label=r'$k_{true} = k_{pred}$')
        ax.errorbar(k_reals, k_predictions, yerr=k_error, fmt='.b', label="test")

        ax.set_xlabel(r"$k_{true}$", fontsize=12)
        ax.set_ylabel(r"$k_{pred}$", fontsize=12)

        fig.suptitle(f'Sequence Length (L) = {configuration.dataset.sequence_length}, Number of Sequences (N)'
                     f' = {configuration.dataset.num_trajectories}', fontsize=14)

        plt.legend(loc='best', fontsize=12)
        plt.show()
        plt.close()


def plot_predictions_mean_std(configuration, root):
    for trajectories in tqdm(tqdm([1 if i == 0 else i * 10 for i in range(17)])):
        configuration.dataset.num_trajectories = trajectories
        configuration.dataset.output_folder = root + '_' + str(trajectories)

        path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

        predictions_file = os.path.join(path, 'predictions.csv')
        df_predictions = pandas.read_csv(predictions_file)

        reals = df_predictions.real.values
        predictions = df_predictions.predictions.values

        k_reals = []
        k_predictions = []
        k_error = []

        for k in numpy.unique(reals):
            reals_k = reals[reals == k]
            predictions_k = predictions[reals == k]

            # mean(log(k_pred/k_true))
            mean = numpy.mean(numpy.log(predictions_k / reals_k))
            std = numpy.std(numpy.log(predictions_k / reals_k))

            k_reals.append(k)
            k_predictions.append(mean)
            k_error.append(std)

        fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))

        ax.plot(k_reals, k_predictions, '--o', label=r'$mean\left(log(\frac{k_{pred}}{k_{true}})\right)$')
        ax.plot(k_reals, k_error, '--o', label=r'$std\left(log(\frac{k_{pred}}{k_{true}})\right)$')
        ax.axhline(y=0, color='k')

        ax.set_xlabel(r"$k_{true}$", fontsize=12)
        ax.set_ylim(-0.1, 0.15)

        fig.suptitle(f'Sequence Length (L) = {configuration.dataset.sequence_length}, Number of Sequences (N)'
                     f' = {configuration.dataset.num_trajectories}', fontsize=14)

        plt.legend(loc='best', fontsize=12)
        plt.show()
        plt.close()


def plot_predictions_pdf(configuration, root):
    for trajectories in tqdm(tqdm([1 if i == 0 else i * 10 for i in range(17)])):
        configuration.dataset.num_trajectories = trajectories
        configuration.dataset.output_folder = root + '_' + str(trajectories)

        path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

        predictions_file = os.path.join(path, 'predictions.csv')
        df_predictions = pandas.read_csv(predictions_file)

        reals = df_predictions.real.values
        predictions = df_predictions.predictions.values

        fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))

        numpy.random.seed(1551)
        ks = sorted(numpy.random.choice(numpy.unique(reals), 4))
        for k in ks:
            reals_k = reals[reals == k]
            predictions_k = predictions[reals == k]

            pdf = numpy.log(predictions_k / reals_k)
            sns.kdeplot(data=pdf, common_norm=True, log_scale=True, ax=ax, label=f"k = {k}")

            ax.set_xlabel(r'$log(\frac{k_{pred}}{k_{true}})$', fontsize=12)
            ax.set_ylabel("PDF", fontsize=12)

            fig.suptitle(f'Sequence Length (L) = {configuration.dataset.sequence_length}, Number of Sequences (N)'
                         f' = {configuration.dataset.num_trajectories}', fontsize=14)

            plt.legend(loc='best', fontsize=12)

        plt.show()
        plt.close()


def confront_models(root, runs, param):
    models = {}

    for run in runs:
        len_of_sequences = []
        mse = []

        os.makedirs(os.path.join(root, 'test_set_error_through_models'), exist_ok=True)
        filename = os.path.join(root, 'test_set_error_through_models', 'test_set_error')

        for sequence in tqdm([128, 256, 512, 1024, 2048]):
            configuration.dataset.sequence_length = sequence
            configuration.dataset.output_folder = root + f'/{run}/{param}_' + str(sequence)

            path = os.path.join(configuration.dataset.output_folder, 'log/v_0')

            test_file = os.path.join(path, 'test.csv')
            df_test_loss = pandas.read_csv(test_file)

            # number_of_sequences.append(trajectories)
            len_of_sequences.append(sequence)
            mse.append(df_test_loss.value.values[0])

        models[run] = mse

    fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))
    ax.plot(len_of_sequences, models[runs[0]], "o", label='1D Convolution')
    ax.plot(len_of_sequences, models[runs[1]], "o", label='Long Short-Term Memory')


    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    ax.set_xlabel('Sequence Length (L)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)

    fig.suptitle(f'Number of Trajectories (N) = {configuration.dataset.num_trajectories}', fontsize=14)

    plt.axhline(y=0.05, color='r', linestyle='--')

    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.90), fontsize=12)

    plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print(f"Usage: {sys.argv[0]} [YAML CONFIGURATION FILE]")
    #     sys.exit(0)
    #
    # path = Path(sys.argv[1])

    path = Path("/home/wonderland/PycharmProjects/Sandard_Map/output/chirikov02_models/run12/configuration.yaml")

    """
    Load Configuration File
    """
    if not path.exists():
        print("Configuration file does not exist! Run again with a valid configuration file!")
        sys.exit(0)

    with path.open('r') as f:
        configuration = edict(yaml.safe_load(f))

    root = os.path.join(configuration.dataset.output_folder, 'chirikov02_models', 'run12', '160')

    """
    Plot Learning Curves
    """
    plot_learning_curves(configuration, root)

    """
    Plot Mean Squared Error on Test Set
    """
    plot_test_mse(configuration, root)

    """
    Plot Predictions RUN
    """
    # plot_predictions(configuration, root)
    # plot_predictions_mean_std(configuration, root)
    # plot_predictions_pdf(configuration, root)

    # confront_models(os.path.join(configuration.dataset.output_folder, 'chirikov02_models'),
    #                 ['run01', 'run05'], '160')

    print("Finished!")
