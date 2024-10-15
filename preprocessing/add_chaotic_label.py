import os
import numpy as np
import pandas as pd

root = '/home/silver/Documents/PycharmProjects/Standard_Map/dataset'

iterations = [128, 256, 512, 1024, 2048]
sequences = [1] + np.arange(10, 170, 10).tolist()

k_values   = [0.6  , 0.75 , 0.9  , 1.1  , 1.3  , 1.55 , 2    , 2.25 , 2.65 , 2.75 , 3.5  , 3.6  , 4    , 4.4  , 4.5  ]
thresholds = [0.05 , 0.05 , 0.015, 0.05 , 0.08 , 0.12 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 ]

for i in iterations:
    for s in sequences:
        # Load p and q data
        p_path = os.path.join(root, 'chirikov02', str(i), str(s), 'test', 'p.csv')
        q_path = os.path.join(root, 'chirikov02', str(i), str(s), 'test', 'q.csv')
        
        p = pd.read_csv(p_path)
        q = pd.read_csv(q_path)

        # Initialize new 'lyapunov_exponent' column with NaN
        p['chaotic'] = -1
        q['chaotic'] = -1

        for k, t in zip(k_values, thresholds):
            mask = np.isclose(q['k'], k, atol=0.001)
            pk = p.loc[mask]
            qk = q.loc[mask]

            # Load the corresponding Lyapunov exponent CSV for this k value
            lyapunov_exponent = pd.read_csv(os.path.join(root, 'lyapunov_exponent', 'test', f'lyapunov_{k:.2f}.csv'))

            # Iterate through matching rows and update lyapunov_exponent column
            for pk0, qk0, idx in zip(pk['0'], qk['0'], pk.index):
                exp = lyapunov_exponent[(lyapunov_exponent['p_initial'] == pk0) & (lyapunov_exponent['q_initial'] == qk0)]['lyapunov_exponent'].values[0]
                label = not(exp <= t)  # True: chaotic False: non chaotic 
                
                # Use .at to safely update specific rows without chaining operations
                p.at[idx, 'chaotic'] = int(label)
                q.at[idx, 'chaotic'] = int(label)

        # Save the modified datasets back to their respective files
        p.to_csv(p_path, index=False)  # Overwrite the original p.csv file
        q.to_csv(q_path, index=False)  # Overwrite the original q.csv file

