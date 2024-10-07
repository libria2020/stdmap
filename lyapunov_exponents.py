import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats


def chirikov(p: np.ndarray, q: np.ndarray, k: float):
    """
    Applies Chirikov's standard map once with parameter k.

    Parameters:
    - p: array-like, momentum values at the current iteration.
    - q: array-like, position values at the current iteration.
    - k: float, nonlinearity parameter for the standard map.

    Returns:
    - p_prime: updated momentum values after one iteration.
    - q_prime: updated position values after one iteration.
    """
    p_prime = (p + k * np.sin(q)) % (2 * np.pi)
    q_prime = (p_prime + q) % (2 * np.pi)
    return p_prime, q_prime


def evolution_with_lyapunov(k: float, r: float, n: int, s: int = None, delta0: float = 1e-8, delta_min=1e-12,
                            ignore=1e3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the Chirikov standard map and calculates Lyapunov exponents to identify chaotic regions.

    Parameters:
    - k: float, nonlinearity parameter for the map.
    - r: float, resolution of the grid in phase space.
    - n: int, number of iterations for the evolution.
    - s: int, optional, number of points to sample from phase space.
    - delta0: float, optional, initial perturbation for Lyapunov calculation.
    - delta_min: float, optional, minimum separation to avoid underflow in Lyapunov calculation.
    - ignore: int, optional, number of initial iterations to ignore in Lyapunov calculation.

    Returns:
    - p: np.ndarray, initial momentum values.
    - q: np.ndarray, initial position values.
    - lyapunov_exponents: np.ndarray, Lyapunov exponents indicating chaotic behavior.
    """

    # Initialize grid of phase space points
    p_init = np.arange(0, 2 * np.pi, r)
    q_init = np.arange(0, 2 * np.pi, r)

    p_it = np.tile(p_init, len(p_init))
    q_it = np.repeat(q_init, len(q_init))

    if s is not None:
        # Randomly sample points if the s parameter is specified
        np.random.seed(0)
        values = sorted(np.random.choice(len(p_it), s, replace=False))
        p_it = p_it[values]
        q_it = q_it[values]

    # Assign colors based on the initial position in phase space
    colours = np.array([p_it / (2 * np.pi), (p_it + q_it) / (4 * np.pi), q_it / (2 * np.pi)]).T

    p = p_it
    q = q_it

    # Perturb the initial conditions slightly to compute Lyapunov exponents
    p_it_perturbed = p_it + delta0
    q_it_perturbed = q_it + delta0

    # Arrays to store Lyapunov exponents
    lyapunov_exponents = np.zeros(len(p_it))

    for i in tqdm(range(n - 1)):
        # Apply the map to both the original and perturbed points
        p_it, q_it = chirikov(p_it, q_it, k=k)  # Apply map to original points
        p_it_perturbed, q_it_perturbed = chirikov(p_it_perturbed, q_it_perturbed, k=k)  # Apply map to perturbed points

        # Compute the separation between original and perturbed trajectories
        delta_p = p_it_perturbed - p_it  # Difference in momentum
        delta_q = q_it_perturbed - q_it  # Difference in position

        # Calculate Euclidean distance (separation)
        delta = np.sqrt(delta_p ** 2 + delta_q ** 2)

        # Ensure delta doesn't drop below delta_min to avoid division by zero
        delta[delta < delta_min] = delta_min

        if i >= ignore:
            # Update the running sum of logarithmic divergences for the Lyapunov exponent
            lyapunov_exponents += np.log(delta / delta0)

        # Renormalize the perturbed points to maintain a constant small separation
        p_it_perturbed = p_it + (delta0 / delta) * delta_p  # Scale the momentum perturbation
        q_it_perturbed = q_it + (delta0 / delta) * delta_q  # Scale the position perturbation

    # Average the Lyapunov exponents over the number of iterations
    lyapunov_exponents = lyapunov_exponents / n

    return p, q, colours, lyapunov_exponents


def compute_threshold(lyapunov_exponents: np.ndarray) -> float:
    """
    Computes the threshold for chaotic trajectories based on Lyapunov exponents.

    Parameters:
    - lyapunov_exponents: np.ndarray, Lyapunov exponents for the trajectories.

    Returns:
    - threshold: float, the computed threshold value.
    """
    # Number of bins using Freedman-Diaconis rule
    N = len(lyapunov_exponents)
    Q3 = np.quantile(lyapunov_exponents, 0.75)
    Q1 = np.quantile(lyapunov_exponents, 0.25)
    IQR = Q3 - Q1
    h = 2 * IQR / pow(N, 1 / 3)  # Bin width
    bins = (np.max(lyapunov_exponents) - np.min(lyapunov_exponents)) / h
    bins = math.ceil(bins)

    # Create histogram
    counts, bin_edges = np.histogram(lyapunov_exponents, bins=bins)

    # Find peaks in the histogram
    peaks, _ = find_peaks(counts)

    # Ensure there are at least two peaks
    if len(peaks) < 2:
        raise ValueError("Less than two peaks found in the histogram.")

    # Get the x-values of the peaks (bin centers corresponding to peaks)
    values = (bin_edges[peaks] + bin_edges[peaks + 1]) / 2

    # Calculate the threshold as the mean between the smallest and highest peaks on the x-axis
    threshold = (np.min(lyapunov_exponents) + np.max(lyapunov_exponents)) / 2
    return threshold


def plot_phase_space(p: np.ndarray, q: np.ndarray, chaotic_trajectories: np.ndarray,
                     non_chaotic_trajectories: np.ndarray):
    """
    Plots the phase space (magnetogram) for chaotic and non-chaotic trajectories.

    Parameters:
    - p: np.ndarray, momentum values.
    - q: np.ndarray, position values.
    - chaotic_trajectories: np.ndarray, boolean mask for chaotic trajectories.
    - non_chaotic_trajectories: np.ndarray, boolean mask for non-chaotic trajectories.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(q[:, non_chaotic_trajectories], p[:, non_chaotic_trajectories], s=1, color='r', alpha=0.15)
    plt.scatter(q[:, chaotic_trajectories], p[:, chaotic_trajectories], s=1, color='b', alpha=0.15)
    plt.title("Chaotic and Non-Chaotic Trajectories (Phase Space)")
    plt.xlabel("q")
    plt.ylabel("p")
    plt.show()


def plot_histogram_with_threshold(lyapunov_exponents: np.ndarray, threshold: float, idx: int):
    """
    Plots the histogram of Lyapunov exponents and adds a vertical line at the threshold.

    Parameters:
    - lyapunov_exponents: np.ndarray, Lyapunov exponents for the trajectories.
    - threshold: float, the threshold value.
    - idx: int, index for saving the plot.
    """
    # Number of bins using Freedman-Diaconis rule
    N = len(lyapunov_exponents)
    Q3 = np.quantile(lyapunov_exponents, 0.75)
    Q1 = np.quantile(lyapunov_exponents, 0.25)
    IQR = Q3 - Q1
    h = 2 * IQR / pow(N, 1 / 3)  # Bin width
    bins = (np.max(lyapunov_exponents) - np.min(lyapunov_exponents)) / h
    bins = math.ceil(bins)

    # Plot the histogram of Lyapunov exponents
    plt.figure(figsize=(10, 6))
    plt.hist(lyapunov_exponents, bins=bins, color='blue', edgecolor='black', alpha=0.7)

    # Adding labels and title
    plt.title('Histogram of Lyapunov Exponents', fontsize=16)
    plt.xlabel('Lyapunov Exponent', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Draw a vertical line at the threshold
    plt.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')

    # Display the legend and grid
    plt.legend()
    plt.grid(True)

    # Save the plot in the "histograms" directory
    plt.savefig(os.path.join('histograms', f'{idx}.png'), dpi=300, bbox_inches='tight')


def save_lyapunov_data(k: float, p: np.ndarray, q: np.ndarray, lyapunov_exponents: np.ndarray, folder: str):
    """
    Saves the initial values of p, q, k, and the Lyapunov exponents to a CSV file.

    Parameters:
    - k: float, the parameter k value.
    - p: np.ndarray, initial momentum values.
    - q: np.ndarray, initial position values.
    - lyapunov_exponents: np.ndarray, the computed Lyapunov exponents.
    - folder: str, the folder where the CSV files will be saved.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(os.path.join(folder, 'data'))

    # Create a DataFrame with the data
    data = {
        'p_initial': p,
        'q_initial': q,
        'k': k,
        'lyapunov_exponent': lyapunov_exponents
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    file_name = os.path.join(folder, f'lyapunov_{k:.2f}.csv')
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    # Create a dictionary with parameters
    params = {
        'k_init': 0,
        'k_end': 5,
        'k_step': 0.05,
        'resolution': 0.5,
        'iterations': 10 ** 9,
        'delta0': 1e-8,
        'delta_min': 1e-12,
        'ignore': 1e3,
        'folder': '../dataset/lyapunov_exponent/01',
    }

    # Create the folder if it doesn't exist
    if not os.path.exists(params['folder']):
        os.makedirs(os.path.join(params['folder']))

    # Writing the dictionary to the txt file
    with open(os.path.join(params['folder'], 'note.txt'), 'w') as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")

    # Generate values of k
    k_values = np.arange(params['k_init'], params['k_end'], params['k_step'])

    for idx, k in enumerate([1.10, 2.65, 4.00]):
    # for idx, k in enumerate(k_values):
        if idx % 10 == 0:
            print(idx)

        # Run the simulation
        p, q, colours, lyapunov_exponents = evolution_with_lyapunov(k, params['resolution'], params['iterations'])

        # Save data to a CSV file
        save_lyapunov_data(k, p, q, lyapunov_exponents, params['folder'])

        # # Compute the threshold
        # threshold = compute_threshold(lyapunov_exponents)
        # print(f"k = {k} - Threshold = {threshold:.2f}")
        #
        # # Filter chaotic and non-chaotic trajectories based on the computed threshold
        # chaotic_trajectories = lyapunov_exponents > threshold
        # non_chaotic_trajectories = lyapunov_exponents <= threshold
        #
        # # Plot the phase space
        # plot_phase_space(p, q, chaotic_trajectories, non_chaotic_trajectories)
        #
        # # Plot the histogram with the threshold
        # plot_histogram_with_threshold(lyapunov_exponents, threshold, idx)


# TODO: cancel from dwnld_data
# TODO: prepared a code that computes in parallel the exponent for all test set
#  [ 1.3, 2.65, 3.5, 0.75, 2.25, 4.4, 2, 0.6, 3.6, 2.75, 4, 0.9, 4.5, 1.55, 1.1 ]
# TODO: redo all figures of the paper with other values of k
