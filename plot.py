import os
import sys

import numpy
import numpy as np
import pandas

import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt, cm, ticker


def chirikov(p: numpy.ndarray, q: numpy.ndarray, k: float):
    """
    Applies Chirikov's standard map once with parameter k
    """

    p_prime = (p + k * numpy.sin(q)) % (2 * numpy.pi)
    q_prime = (p + q + k * numpy.sin(q)) % (2 * numpy.pi)

    return p_prime, q_prime


def evolution(k: float, r: float, n: int, s: int = None) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    generates dataset for N iterations of the Chirikov map with parameter k and resolution r
    constant colour represents the same original point but after multiple iterations
    """

    p_init = numpy.arange(0, 2 * numpy.pi, r)
    q_init = numpy.arange(0, 2 * numpy.pi, r)

    p_it = numpy.tile(p_init, len(p_init))
    q_it = numpy.repeat(q_init, len(q_init))

    if s is not None:
        numpy.random.seed(0)
        values = sorted(numpy.random.choice(len(p_it), s, replace=False))
        p_it = p_it[values]
        q_it = q_it[values]

    """
    The original colour of a point is uniquely determined by an RGB value determined by
    the starting point where Red is given by p_init, Blue is given by q_init, and
    Green is given by p_init+q_init.
    """
    colours = numpy.array([p_it / (2 * numpy.pi), (p_it + q_it) / (4 * numpy.pi), q_it / (2 * numpy.pi)]).T

    p = p_it
    q = q_it

    for i in tqdm(range(n - 1)):
        p_it, q_it = chirikov(p_it, q_it, k=k)

        p = numpy.vstack((p, p_it))
        q = numpy.vstack((q, q_it))

    return p, q, colours


def plot(k: int, p: numpy.ndarray, q: numpy.ndarray, colours: numpy.ndarray, directory: str) -> None:
    filename = os.path.join(directory, "k_test_%.2f" % k)

    cm = 'magma'

    plt.clf()
    plt.scatter(q, p, s=.15, linewidths=.15, edgecolors=None, c=colours[:, 1], cmap=cm)

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')
    plt.title("k = %.2f" % k)

    plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # # plot one
    # k = 20
    # p, q, colours = evolution(k, 0.5, 4096)
    # colours = numpy.vstack([colours] * 4096)
    # cm = 'magma'
    # fontdict = {'fontsize': 12}
    #
    # fig, axs = plt.subplots(figsize=(7, 7), layout='constrained')
    # axs.scatter(q, p, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    #
    # axs.set_ylabel(r'$p$')
    # axs.set_xlabel(r'$\theta$')
    #
    # axs.tick_params(axis='x', labelsize=8)
    # axs.tick_params(axis='y', labelsize=8)
    # axs.tick_params(length=2, grid_alpha=0.5)
    # axs.set_title(r"$k = %.2f$" % k, fontdict=fontdict)
    #
    # plt.savefig(f'{k}.png', format='png', dpi=300, bbox_inches='tight')
    #
    # # plot four
    # p_0, q_0, colours = evolution(0.5, 0.5, 2048)
    # p_1, q_1, colours = evolution(1.3, 0.5, 2048)
    # p_2, q_2, colours = evolution(2.1, 0.5, 2048)
    # p_3, q_3, colours = evolution(5, 0.5, 2048)
    #
    # colours = numpy.vstack([colours] * 2048)
    # filename = os.path.join('plots/phase_space/', "fig")
    #
    # k_values = [0.5, 1.3, 2.1, 5]
    #
    # fontdict = {'fontsize': 12}
    #
    # cm = 'magma'
    #
    # fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(10, 10), sharex=True, sharey=True)
    #
    # axs[0, 0].scatter(q_0, p_0, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    # axs[0, 1].scatter(q_1, p_1, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    # axs[1, 0].scatter(q_2, p_2, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    # axs[1, 1].scatter(q_3, p_3, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    #
    # axs[0, 0].set_ylabel(r'$p$')
    # axs[1, 0].set_ylabel(r'$p$')
    #
    # axs[1, 0].set_xlabel(r'$\theta$')
    # axs[1, 1].set_xlabel(r'$\theta$')
    #
    # for i in range(2):
    #     for j in range(2):
    #         axs[i, j].tick_params(axis='x', labelsize=8)
    #         axs[i, j].tick_params(axis='y', labelsize=8)
    #         axs[i, j].tick_params(length=2, grid_alpha=0.5)
    #         axs[i, j].set_title(r"$k = %.2f$" % k_values[i * 2 + j], fontdict=fontdict)
    #
    # plt.savefig('fs.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fs.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # plt.show()
    #
    # # k = 8
    #
    # for n in [512, 1024, 2048]:
    #     for s in [40, 80, 120]:
    #         p, q, colours = evolution(k, 0.5, n, s)
    #
    #         colours = numpy.vstack([colours] * n)
    #         filename = os.path.join('plots/phase_space/3', f'k_{k: .2f}_{n}_{s}')
    #
    #         fontdict = {'fontsize': 12}
    #         cm = 'magma'
    #         fig, axs = plt.subplots(layout='constrained')
    #
    #         axs.scatter(q, p, s=.15, linewidths=.15, c=colours[:, 1], cmap=cm)
    #
    #         axs.set_ylabel(r'$p$')
    #         axs.set_xlabel(r'$\theta$')
    #
    #         axs.tick_params(axis='x', labelsize=8)
    #         axs.tick_params(axis='y', labelsize=8)
    #         axs.tick_params(length=2, grid_alpha=0.5)
    #         axs.set_title(r"$K = %.2f$, $L = %d$, $N = %d$ " % (k, n, s), fontdict=fontdict)
    #
    #         plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
    #         plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    #         plt.show()

    # ---------------------------------------------------------------------------------------------------------------- #

    # for k in [1.2, 2.2, 5.3, 8]:
    #
    #     filename = os.path.join('plots/phase_space/', f"phase_space_{k}")
    #
    #     p_0, q_0, colours_0 = evolution(k, 0.5, 512, 40)
    #     p_1, q_1, colours_1 = evolution(k, 0.5, 512, 80)
    #     p_2, q_2, colours_2 = evolution(k, 0.5, 512, 120)
    #     p_3, q_3, colours_3 = evolution(k, 0.5, 1024, 40)
    #     p_4, q_4, colours_4 = evolution(k, 0.5, 1024, 80)
    #     p_5, q_5, colours_5 = evolution(k, 0.5, 1024, 120)
    #     p_6, q_6, colours_6 = evolution(k, 0.5, 2048, 40)
    #     p_7, q_7, colours_7 = evolution(k, 0.5, 2048, 80)
    #     p_8, q_8, colours_8 = evolution(k, 0.5, 2048, 120)
    #
    #     colours_0 = numpy.vstack([colours_0] * 512)
    #     colours_1 = numpy.vstack([colours_1] * 512)
    #     colours_2 = numpy.vstack([colours_2] * 512)
    #
    #     colours_3 = numpy.vstack([colours_3] * 1024)
    #     colours_4 = numpy.vstack([colours_4] * 1024)
    #     colours_5 = numpy.vstack([colours_5] * 1024)
    #
    #     colours_6 = numpy.vstack([colours_6] * 2048)
    #     colours_7 = numpy.vstack([colours_7] * 2048)
    #     colours_8 = numpy.vstack([colours_8] * 2048)
    #
    #     fontdict = {'fontsize': 10}
    #     cm = 'magma'
    #
    #     fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(9, 9))
    #
    #     axs[0, 0].scatter(q_0, p_0, s=.15, linewidths=.15, c=colours_0[:, 1], cmap=cm)
    #     axs[0, 1].scatter(q_1, p_1, s=.15, linewidths=.15, c=colours_1[:, 1], cmap=cm)
    #     axs[0, 2].scatter(q_2, p_2, s=.15, linewidths=.15, c=colours_2[:, 1], cmap=cm)
    #     axs[1, 0].scatter(q_3, p_3, s=.15, linewidths=.15, c=colours_3[:, 1], cmap=cm)
    #     axs[1, 1].scatter(q_4, p_4, s=.15, linewidths=.15, c=colours_4[:, 1], cmap=cm)
    #     axs[1, 2].scatter(q_5, p_5, s=.15, linewidths=.15, c=colours_5[:, 1], cmap=cm)
    #     axs[2, 0].scatter(q_6, p_6, s=.15, linewidths=.15, c=colours_6[:, 1], cmap=cm)
    #     axs[2, 1].scatter(q_7, p_7, s=.15, linewidths=.15, c=colours_7[:, 1], cmap=cm)
    #     axs[2, 2].scatter(q_8, p_8, s=.15, linewidths=.15, c=colours_8[:, 1], cmap=cm)
    #
    #     axs[0, 0].set_ylabel(r'$p$')
    #     axs[1, 0].set_ylabel(r'$p$')
    #     axs[2, 0].set_ylabel(r'$p$')
    #
    #     axs[2, 0].set_xlabel(r'$\theta$')
    #     axs[2, 1].set_xlabel(r'$\theta$')
    #     axs[2, 2].set_xlabel(r'$\theta$')
    #
    #     axs[0, 0].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #     axs[0, 1].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #     axs[0, 2].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #
    #     axs[1, 0].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #     axs[1, 1].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #     axs[1, 2].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #
    #     axs[0, 1].yaxis.set_tick_params(left=False, labelleft=False)
    #     axs[0, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    #     axs[1, 1].yaxis.set_tick_params(left=False, labelleft=False)
    #     axs[1, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    #     axs[2, 1].yaxis.set_tick_params(left=False, labelleft=False)
    #     axs[2, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    #     for i, n in enumerate([512, 1024, 2048]):
    #         for j, s in enumerate([40, 80, 120]):
    #             axs[i, j].tick_params(axis='x', labelsize=8)
    #             axs[i, j].tick_params(axis='y', labelsize=8)
    #             axs[i, j].tick_params(length=2, grid_alpha=0.5)
    #             axs[i, j].set_title(r"$L = %d$, $N = %d$ " % (n, s), fontdict=fontdict)
    #
    #     fig.suptitle(r"$K = %.2f$ " % k, fontsize=14)
    #
    #     # plt.savefig(filename + '.png', format='png', dpi=300, bbox_inches='tight')
    #     # plt.savefig(filename + '.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    #     plt.show()

    # ---------------------------------------------------------------------------------------------------------------- #

    # k = 5.3
    # p_0, q_0, colours_0 = evolution(k, 0.5, 512, 100)
    # p_1, q_1, colours_1 = evolution(k, 0.5, 512, 110)
    # colours_0 = numpy.vstack([colours_0] * 512)
    # colours_1 = numpy.vstack([colours_1] * 512)
    #
    # fontdict = {'fontsize': 10}
    # cm = 'magma'
    #
    # fig, axs = plt.subplots(layout='constrained', figsize=(9, 9))
    #
    # axs.scatter(q_1, p_1, s=.15, linewidths=.5, c='red')
    # axs.scatter(q_0, p_0, s=.15, linewidths=.5, c='black')
    #
    # axs.set_ylabel(r'$p$')
    # axs.set_xlabel(r'$\theta$')
    #
    # axs.tick_params(axis='x', labelsize=8)
    # axs.tick_params(axis='y', labelsize=8)
    # axs.tick_params(length=2, grid_alpha=0.5)
    # axs.set_title(r"$L = %d$, $N = %d$ " % (512, 100), fontdict=fontdict)
    #
    # plt.show()

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #

    # root = '/home/silver/Documents/PycharmProjects/stdmap/dataset/chirikov02/'
    #
    # dfp = pandas.read_csv("/home/silver/Documents/PycharmProjects/stdmap/dataset/chirikov02/512/1/test/p.csv")
    # k = dfp['k'].values[-1]
    # print(k)
    #
    # fontdict = {'fontsize': 12}
    # cm = 'magma'
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, n in enumerate([128, 512, 2048]):
    #     for j, s in enumerate([50, 100, 150]):
    #         p_file = os.path.join(root, str(n), str(s), 'test/p.csv')
    #         q_file = os.path.join(root, str(n), str(s), 'test/q.csv')
    #
    #         dfp = pandas.read_csv(p_file)
    #         dfq = pandas.read_csv(q_file)
    #
    #         p = dfp.loc[dfq['k'] == k].values[:, :-1].T
    #         q = dfq.loc[dfq['k'] == k].values[:, :-1].T
    #
    #         p_it = dfp.loc[dfq['k'] == k].values[:, 0].T
    #         q_it = dfq.loc[dfq['k'] == k].values[:, 0].T
    #
    #         colours = numpy.array([p_it / (2 * numpy.pi), (p_it + q_it) / (4 * numpy.pi), q_it / (2 * numpy.pi)]).T
    #         colours = numpy.vstack([colours] * n)
    #
    #         axs[i, j].scatter(q, p, s=.15, linewidths=.5, c=colours[:, 1], cmap=cm)
    #
    # axs[0, 0].set_ylabel(r'$p$')
    # axs[1, 0].set_ylabel(r'$p$')
    # axs[2, 0].set_ylabel(r'$p$')
    #
    # axs[2, 0].set_xlabel(r'$\theta$')
    # axs[2, 1].set_xlabel(r'$\theta$')
    # axs[2, 2].set_xlabel(r'$\theta$')
    #
    # axs[0, 0].xaxis.set_tick_params(bottom=False, labelbottom=False)
    # axs[0, 1].xaxis.set_tick_params(bottom=False, labelbottom=False)
    # axs[0, 2].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #
    # axs[1, 0].xaxis.set_tick_params(bottom=False, labelbottom=False)
    # axs[1, 1].xaxis.set_tick_params(bottom=False, labelbottom=False)
    # axs[1, 2].xaxis.set_tick_params(bottom=False, labelbottom=False)
    #
    # axs[0, 1].yaxis.set_tick_params(left=False, labelleft=False)
    # axs[0, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    # axs[1, 1].yaxis.set_tick_params(left=False, labelleft=False)
    # axs[1, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    # axs[2, 1].yaxis.set_tick_params(left=False, labelleft=False)
    # axs[2, 2].yaxis.set_tick_params(left=False, labelleft=False)
    #
    # fig.suptitle(r"$k = %.2f$ " % k, fontsize=14)
    #
    # for i, n in enumerate([512, 1024, 2048]):
    #     for j, s in enumerate([40, 80, 120]):
    #         axs[i, j].tick_params(axis='x', labelsize=8)
    #         axs[i, j].tick_params(axis='y', labelsize=8)
    #         axs[i, j].tick_params(length=2, grid_alpha=0.5)
    #         axs[i, j].set_title(r"$L = %d$, $N = %d$ " % (n, s), fontdict=fontdict)
    #
    # plt.savefig('fig.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig.eps', format='eps', dpi=300, bbox_inches='tight')
    #

    # # ---------------------------------------------------------------------------------------------------------------- #
    # fontdict = {'fontsize': 12}
    #
    # sequences = [128, 512, 2048]
    # sequences_folder = ['run05', 'run07', 'run09']
    # trajectories = [50, 100, 150]
    #
    # root = '/home/silver/Documents/PycharmProjects/Standard_Map/output/chirikov02v1/lstm'

    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # PDF
    # """
    #
    # k_values = [1.1, 2.65, 4.0]
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, k in enumerate(k_values):
    #     for j, (l, sl) in enumerate(zip(sequences, sequences_folder)):
    #         for n in trajectories:
    #             path = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/predictions.csv')
    #
    #             df = pandas.read_csv(path)
    #
    #             mask = numpy.isclose(df['real'], k, atol=0.001)
    #             filtered_predictions = df.loc[mask]
    #
    #             real = filtered_predictions['real'].values
    #             predictions = filtered_predictions['predictions'].values
    #
    #             pdf = np.log(predictions / k)
    #
    #             sns.kdeplot(data=pdf, ax=axs[i, j], label=f"{n} trajectories")
    #
    #             axs[i, j].set_title(f"k = {k:.2f}, L={l}", fontdict=fontdict)
    #             axs[i, j].set_xlabel(r'$log(\frac{k_{pred}}{k_{true}})$', fontdict=fontdict)
    #             axs[i, j].set_ylabel("PDF", fontdict=fontdict)
    #
    #             # axs[i, j].set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    #             # axs[i, j].set_xticklabels(['0.8', '0.9', '1.0', '1.1', '1.2'])
    #
    #             axs[i, j].legend()
    #
    # plt.xlim([-0.4, 0.4])
    #
    # plt.savefig('fig1.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig1.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, k in enumerate(k_values):
    #     for j, n in enumerate(trajectories):
    #
    #         for idx, (l, sl) in enumerate(zip(sequences, sequences_folder)):
    #             path = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/predictions.csv')
    #
    #             df = pandas.read_csv(path)
    #
    #             mask = numpy.isclose(df['real'], k, atol=0.001)
    #             filtered_predictions = df.loc[mask]
    #
    #             real = filtered_predictions['real'].values
    #             predictions = filtered_predictions['predictions'].values
    #
    #             pdf = np.log(predictions / k)
    #
    #             sns.kdeplot(data=pdf, ax=axs[i, j], label=f"{l} iterations")
    #
    #             axs[i, j].set_title(f"k = {k:.2f}, N={n}", fontdict=fontdict)
    #             axs[i, j].set_xlabel(r'$log(\frac{k_{pred}}{k_{true}})$', fontdict=fontdict)
    #             axs[i, j].set_ylabel("PDF", fontdict=fontdict)
    #
    #             # axs[i, j].set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    #             # axs[i, j].set_xticklabels(['0.8', '0.9', '1.0', '1.1', '1.2'])
    #
    #             axs[i, j].legend()
    #
    # plt.xlim([-0.4, 0.4])
    #
    # plt.savefig('fig2.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig2.eps', format='eps', dpi=300, bbox_inches='tight')

    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Error
    # """
    #
    # k_values = [0.6, 0.75, 0.9, 1.1, 1.3, 1.55, 2.0, 2.25, 2.65, 2.75, 3.5, 4.0, 4.4, 4.5]
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, (l, sl) in enumerate(zip(sequences, sequences_folder)):
    #     for j, n in enumerate(trajectories):
    #         path = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/predictions.csv')
    #
    #         df = pandas.read_csv(path)
    #
    #         mean = []
    #         std = []
    #
    #         for k in k_values:
    #             mask = numpy.isclose(df['real'], k, atol=0.001)
    #             filtered_predictions = df.loc[mask]
    #
    #             real = filtered_predictions['real'].values
    #             predictions = filtered_predictions['predictions'].values
    #
    #             mean.append(numpy.mean(predictions))
    #             std.append(numpy.std(predictions))
    #
    #         seqs = [i for i in range(0, 6, 1)]
    #
    #         axs[i, j].plot(seqs, seqs, "k-", label=r'$k_{true} = k_{pred}$')
    #         axs[i, j].errorbar(k_values, mean, yerr=std, fmt='.b', label="test")
    #
    #         axs[i, j].set_title(f"$L={l}, N={n}$", fontdict=fontdict)
    #         axs[2, j].set_xlabel(r"$k_{true}$", fontdict=fontdict)
    #         axs[i, 0].set_ylabel(r"$k_{pred}$", fontdict=fontdict)
    #
    #         axs[i, j].legend()
    #
    # plt.savefig('fig3.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig3.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, (l, sl) in enumerate(zip(sequences, sequences_folder)):
    #     for j, n in enumerate(trajectories):
    #         path = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/predictions.csv')
    #
    #         df = pandas.read_csv(path)
    #
    #         mean = []
    #         std = []
    #
    #         for k in k_values:
    #             mask = numpy.isclose(df['real'], k, atol=0.001)
    #             filtered_predictions = df.loc[mask]
    #
    #             real = filtered_predictions['real'].values
    #             predictions = filtered_predictions['predictions'].values
    #
    #             pdf = numpy.log(predictions / k)
    #
    #             mean.append(numpy.mean(pdf))
    #             std.append(numpy.std(pdf))
    #
    #         axs[2, j].set_xlabel(r'$k_{true}$', fontdict=fontdict)
    #
    #         axs[i, j].plot(k_values, mean, '*-', label=r'$mean\left(log(\frac{k_{pred}}{k_{true}})\right)$')
    #         axs[i, j].plot(k_values, std, '+-', label=r'$std\left(log(\frac{k_{pred}}{k_{true}})\right)$')
    #         axs[i, j].axhline(y=0, color='k')
    #
    #         axs[i, j].set_title(f"$L={l}, N={n}$", fontdict=fontdict)
    #
    #         axs[i, j].legend()
    #
    # plt.savefig('fig4.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig4.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Learning Curves
    # """
    #
    # fig, axs = plt.subplots(3, 3, layout='constrained', figsize=(15, 15), sharex=True, sharey=True)
    #
    # for i, (l, sl) in enumerate(zip(sequences, sequences_folder)):
    #     for j, n in enumerate(trajectories):
    #         path_train = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/train.csv')
    #         path_validation = os.path.join(root, sl, f'{l}_{n}', 'log/v_0/validation.csv')
    #
    #         df_train = pandas.read_csv(path_train)
    #         df_validation = pandas.read_csv(path_validation)
    #
    #         epochs = df_train["epoch"].values
    #         train = df_train["value"].values
    #         validation = df_validation["value"].values
    #
    #         axs[i, j].plot(epochs, train, label='train')
    #         axs[i, j].plot(epochs, validation, label='validation')
    #
    #         axs[i, j].set_title(f"$L={l}, N={n}$", fontdict=fontdict)
    #         axs[2, j].set_xlabel("epochs", fontdict=fontdict)
    #         axs[i, 0].set_ylabel("SmoothL1", fontdict=fontdict)
    #
    #         axs[i, j].legend()
    #
    # plt.savefig('fig5.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig5.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Test Error Two Models
    # """
    #
    # sequences = [128, 256, 512, 1024, 2048]
    # sequences_folder = ['cnn', 'lstm']
    # trajectories = 160
    #
    # root = '/home/silver/Documents/PycharmProjects/Standard_Map/output/chirikov02models'
    #
    # mse = []
    #
    # for sf in sequences_folder:
    #     smse = []
    #     for s in sequences:
    #         path = os.path.join(root, sf, str(trajectories) + '_' + str(s), 'log/v_0/test.csv')
    #
    #         df = pandas.read_csv(path)
    #
    #         smse.append(df.value.values[0])
    #
    #     mse.append(smse)
    #
    # fig, ax = plt.subplots(layout='constrained', figsize=(8, 7))
    # ax.plot(sequences, mse[0], "--o", label='Convolution')
    # ax.plot(sequences, mse[1], "--o", label='LSTM')
    #
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=16))
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #
    # ax.set_yscale('log')
    # ax.set_xscale('log', base=2)
    #
    # ax.set_xlabel('Number of Sequences (N)', fontsize=12)
    # ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    #
    # fig.suptitle(f'Number of Trajectories (N) = 160', fontsize=12)
    #
    # plt.legend(loc='upper right', fontsize=12)
    #
    # plt.savefig('fig6.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig6.eps', format='eps', dpi=300, bbox_inches='tight')
    #
    # # ---------------------------------------------------------------------------------------------------------------- #
    #
    # """
    # Test Error Three Loss Functions
    # """
    #
    # sequences = [128, 256, 512, 1024, 2048]
    # sequences_folder = ['run00', 'run01', 'run02', 'run03', 'run04', 'run05']
    # trajectories = 160
    #
    # root = '/home/silver/Documents/PycharmProjects/Standard_Map/output/chirikovloss'
    #
    # mse = []
    #
    # for sf in sequences_folder:
    #     smse = []
    #     for s in sequences:
    #         path = os.path.join(root, sf, str(trajectories) + '_' + str(s), 'log/v_0/test.csv')
    #
    #         df = pandas.read_csv(path)
    #
    #         smse.append(df.value.values[0])
    #
    #     mse.append(smse)
    # print(mse)
    #
    #
    # fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(12, 6), sharex=True, sharey=True)
    #
    # axs[0].plot(sequences, mse[0], "--o", label='Mean Squared Error (MSE)')
    # axs[0].plot(sequences, mse[1], "--o", label='Mean Absolute Error (L1)')
    # axs[0].plot(sequences, mse[2], "--o", label='Smooth L1')
    #
    # axs[1].plot(sequences, mse[3], "--o", label='Mean Squared Error (MSE)')
    # axs[1].plot(sequences, mse[4], "--o", label='Mean Absolute Error (L1)')
    # axs[1].plot(sequences, mse[5], "--o", label='Smooth L1')
    #
    # axs[0].set_xlabel('Number of Iterations (L)', fontsize=12)
    # axs[1].set_xlabel('Number of Iterations (L)', fontsize=12)
    #
    # axs[0].set_yscale('log')
    # axs[1].set_yscale('log')
    #
    # axs[0].set_xscale('log', base=2)
    # axs[1].set_xscale('log', base=2)
    #
    # plt.ylim([0.004, .1])
    #
    # # axs[0].set_yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    # # axs[0].set_yticklabels(['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'])
    #
    # axs[0].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=16))
    # axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # axs[1].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=16))
    #
    # fig.suptitle(f'Number of Trajectories (N) = 160', fontsize=12)
    #
    # axs[0].title.set_text('Convolutional Neural Network')
    # axs[1].title.set_text('Long-Short Term Memory Network')
    #
    # axs[0].legend(loc='upper right', fontsize=12)
    # axs[1].legend(loc='upper right', fontsize=12)
    #
    # plt.savefig('fig7.png', format='png', dpi=300, bbox_inches='tight')
    # plt.savefig('fig7.eps', format='eps', dpi=300, bbox_inches='tight')

    print("Finished")