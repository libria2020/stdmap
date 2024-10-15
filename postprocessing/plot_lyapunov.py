import os
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    root = '/home/silver/Documents/PycharmProjects/Standard_Map/dataset/lyapunov_exponent/test'

    for k in os.listdir(root):
        print(k[:-4] + '.png')
        if k.endswith('.csv'):
            df = pd.read_csv(os.path.join(root, k))
            lyapunov_exponents = df['lyapunov_exponent'].values
            k_value = df['k'].values[0]

            # Number of bins using Freedman-Diaconis rule
            N = len(lyapunov_exponents)
            Q3 = np.quantile(lyapunov_exponents, 0.75)
            Q1 = np.quantile(lyapunov_exponents, 0.25)
            IQR = Q3 - Q1
            h = 2 * IQR / pow(N, 1 / 3)  # Bin width
            bins = (np.max(lyapunov_exponents) - np.min(lyapunov_exponents)) / h
            bins = math.ceil(bins)
            bins = 100 if bins > 1500 else bins

            # Plot the histogram of Lyapunov exponents
            plt.figure(figsize=(10, 6))
            plt.hist(lyapunov_exponents, bins=bins, color='blue', edgecolor='black', alpha=0.7)

            # Plot the histogram of Threshold
            # plt.axvline(0.05, color='orange', linestyle='--', linewidth=2)

            # Adding labels and title
            plt.title(f'Histogram of Lyapunov Exponents (k={k_value})', fontsize=16)
            plt.xlabel('Lyapunov Exponent', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)

            # Display the grid
            plt.grid(True)

            plt.savefig(k[:-4] + '.png', format='png', dpi=300, bbox_inches='tight')
