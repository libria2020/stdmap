import os
import sys
import datetime
import numpy
import pandas

from dataclasses import dataclass
from argparse import ArgumentParser

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def chirikov(p: numpy.ndarray, q: numpy.ndarray, k: float):
    """
    Applies Chirikov's standard map once with parameter k
    """

    p_prime = (p + k * numpy.sin(q)) % (2 * numpy.pi)
    q_prime = (p + q + k * numpy.sin(q)) % (2 * numpy.pi)

    return p_prime, q_prime


def evolution(k: float, r: float, n: int, s: int = None) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    generates dataset for N iterations of the Chirikov map with parameter k and resolution r

    :param k: value of K
    :param r: resolution of initial conditions (from 0 to 2 pi with step r)
    :param n: number of iterations
    :param s: number of trajectories
    :return: (P, Q) each of  shape (n, s)
    """

    # initial conditions
    p_init = numpy.arange(0, 2 * numpy.pi, r)
    q_init = numpy.arange(0, 2 * numpy.pi, r)

    p_it = numpy.tile(p_init, len(p_init))
    q_it = numpy.repeat(q_init, len(q_init))

    # randomly select `s` number of initial conditions
    if s is not None:
        numpy.random.seed(datetime.datetime.now().microsecond)
        values = sorted(numpy.random.choice(len(p_it), s, replace=False))
        p_it = p_it[values]
        q_it = q_it[values]

    # generate `s` trajectories
    p = numpy.zeros((len(p_it), n))
    q = numpy.zeros((len(q_it), n))

    p[:, 0] = p_it
    q[:, 0] = q_it

    for i in range(1, n):
        p_it, q_it = chirikov(p_it, q_it, k=k)

        p[:, i] = p_it
        q[:, i] = q_it

    return p, q


def create_dataset(k_set, resolution, iterations, trajectories):
    p_set = numpy.empty((0, iterations + 1))
    q_set = numpy.empty((0, iterations + 1))

    for k in k_set:
        p, q = evolution(k, resolution, iterations, trajectories)

        p = numpy.concatenate((p, numpy.full((trajectories, 1), k)), axis=1)
        q = numpy.concatenate((q, numpy.full((trajectories, 1), k)), axis=1)

        p_set = numpy.concatenate((p_set, p), axis=0)
        q_set = numpy.concatenate((q_set, q), axis=0)

    return p_set, q_set


def save_dataset(p_set, q_set, iterations, path):
    columns = [str(k) for k in range(iterations)]
    columns.append('k')

    pdf = pandas.DataFrame(p_set, columns=columns)
    qdf = pandas.DataFrame(q_set, columns=columns)

    try:
        pdf.to_csv(os.path.join(path, 'p.csv'), index=False)
    except OSError as e:
        print(e)

    try:
        qdf.to_csv(os.path.join(path, 'q.csv'), index=False)
    except OSError as e:
        print(e)


@dataclass
class Args:
    dataset_name: str
    resolution: float
    iterations: int
    k_init: int
    k_end: int
    k_step: float

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':

    root = '../dataset'

    parser = ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--resolution", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--k-init", type=int, required=True)
    parser.add_argument("--k-end", type=int, required=True)
    parser.add_argument("--k-step", type=float, required=True)

    args = Args(**parser.parse_args().__dict__)

    # generate values of k
    k_values = numpy.arange(args.k_init, args.k_end, args.k_step)

    # divide values of k in train/validation/test set
    k_train, k_test = train_test_split(k_values, test_size=0.3, random_state=42)
    k_test, k_valid = train_test_split(k_test, test_size=0.5, random_state=42)

    num_init = len(numpy.arange(0, 2 * numpy.pi, args.resolution)) ** 2 // 10 + 1

    for trajectories in tqdm([1 if i == 0 else i * 10 for i in range(num_init)]):
        for split, k_set in zip(['train', 'validation', 'test'], [k_train, k_valid, k_test]):
            # create folder
            path = os.path.join(root, args.dataset_name, str(args.iterations), str(trajectories), split)
            os.makedirs(path, exist_ok=True)

            # compute train/validation/test dataset
            p_set, q_set = create_dataset(k_set, args.resolution, args.iterations, trajectories)

            # save train/validation/test dataset
            save_dataset(p_set, q_set, args.iterations, path)

            # compute mean and standard deviation
            if split == 'train':
                mean_std = pandas.DataFrame({
                    'mean': [numpy.mean(p_set), numpy.mean(q_set)],
                    'std': [numpy.std(p_set), numpy.std(q_set)]
                }, index=['p', 'q'])

                # save mean and standard deviation
                try:
                    mean_std.to_csv(os.path.join(path, 'mean_std.csv'), index=True)
                except OSError as e:
                    print(e)

    print("Finished!")
