# Data Generation

\[
p_{n+1} = p_n + k \sin(q_n)
\]

\[
q_{n+1} = (q_n + p_{n+1}) \mod (2\pi)
\]

To generate the dataset run the fallowing command:

`python3 generate_dataset.py --dataset-name='stdmap' --resolution=0.5 --iterations=128 --k-init=0 --k-end=5 --k-step=0.05`

Afterward, re-run the code, changing `--iterations` to 256, 512, 1024, and 2048 in order to generate the complete 
dataset.

This creates the folder `stdmap/` inside `../dataset/` with all the data required to train our deep learning model. 
Inside `../dataset/stdmap/`, there is a folder for each iteration, and inside each of these folders, we will find 17 
other folders, one for each number of initial conditions investigated. For example, the folder 
`../dataset/stdmap/1024/50/` contains a run of the standard map that, starting with 50 initial conditions, performs 
1024 iterations, generating the corresponding 50 trajectories.

Finally, inside each of these folders, there are 3 other folders: `train`, `validation`, and `test`, which contain 
the values of the phase space variables `p` and `q` in `csv` format. The rows represent the initial conditions and the 
columns represent the number of iterations. The last column contains the `k` value, which is the value we want to 
predict with our deep learning model. Additionally, the train folder also contains the mean and standard deviation 
computed from the training set in `csv` format.

This is an example of the filesystem create by the code:

```angular2html
../dataset/stdmap/
└── 128
    ├── 1
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 10
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 100
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 110
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 120
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 130
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 140
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 150
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 160
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 20
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 30
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 40
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 50
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 60
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 70
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    ├── 80
    │   ├── test
    │   │   ├── p.csv
    │   │   └── q.csv
    │   ├── train
    │   │   ├── mean_std.csv
    │   │   ├── p.csv
    │   │   └── q.csv
    │   └── validation
    │       ├── p.csv
    │       └── q.csv
    └── 90
        ├── test
        │   ├── p.csv
        │   └── q.csv
        ├── train
        │   ├── mean_std.csv
        │   ├── p.csv
        │   └── q.csv
        └── validation
            ├── p.csv
            └── q.csv

```

# Train the deep learning model
For each pair of \(N\) (number of initial conditions generating the respective trajectories) and \(L\) (iterations or 
sequence length), a different model is trained. To train the model, run:

`python3 main.py 'configuration/cnn.yaml'`

for training the convolutional model, and

`python3 main.py 'configuration/lstm.yaml'`

for training the LSTM model.

To select the directory of the different datasets, modify the following line in the configuration file:

```angular2html
dataset:
  input_folder: "../dataset/stdmap"
```

Ensure this matches the name specified with the `--dataset-name` argument when running `generate_dataset.p`y.

To select the number of iterations (128, 256, 512, 1024, 2048), modify the following line in the configuration to train 
the models with all datasets:

```angular2html
dataset:
  sequence_length: 128
```

Internally, we loop through the different values of \(N\) and each time, we train the model from scratch
(e.g., `../dataset/stdmap/512/70/` trains a model with data generated with \(N=70\) and \(L=512\)), 

The output of the training will be saved automatically in `../output/`. You can set another directory by modifying the 
following line in the configuration file:

```angular2html
dataset:
  output_folder: "../output"
```

This is an example of the filesystem create by the code:

```angular2html
../output/
├── 512_1
│   ├── best_model
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712847994.wonderland.365665.0
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712847995.wonderland.365665.1
│           └── loss_validation loss
│               └── events.out.tfevents.1712847995.wonderland.365665.2
├── 512_10
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848013.wonderland.365665.3
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848014.wonderland.365665.4
│           └── loss_validation loss
│               └── events.out.tfevents.1712848014.wonderland.365665.5
├── 512_100
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712849109.wonderland.365665.30
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712849112.wonderland.365665.31
│           └── loss_validation loss
│               └── events.out.tfevents.1712849112.wonderland.365665.32
├── 512_110
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712849337.wonderland.365665.33
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712849342.wonderland.365665.34
│           └── loss_validation loss
│               └── events.out.tfevents.1712849342.wonderland.365665.35
├── 512_120
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712849584.wonderland.365665.36
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712849589.wonderland.365665.37
│           └── loss_validation loss
│               └── events.out.tfevents.1712849589.wonderland.365665.38
├── 512_130
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712849882.wonderland.365665.39
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712849888.wonderland.365665.40
│           └── loss_validation loss
│               └── events.out.tfevents.1712849888.wonderland.365665.41
├── 512_140
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712850220.wonderland.365665.42
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712850226.wonderland.365665.43
│           └── loss_validation loss
│               └── events.out.tfevents.1712850226.wonderland.365665.44
├── 512_150
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712850580.wonderland.365665.45
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712850586.wonderland.365665.46
│           └── loss_validation loss
│               └── events.out.tfevents.1712850586.wonderland.365665.47
├── 512_160
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712850963.wonderland.365665.48
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712850969.wonderland.365665.49
│           └── loss_validation loss
│               └── events.out.tfevents.1712850969.wonderland.365665.50
├── 512_20
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848051.wonderland.365665.6
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848052.wonderland.365665.7
│           └── loss_validation loss
│               └── events.out.tfevents.1712848052.wonderland.365665.8
├── 512_30
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848110.wonderland.365665.9
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848111.wonderland.365665.10
│           └── loss_validation loss
│               └── events.out.tfevents.1712848111.wonderland.365665.11
├── 512_40
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848190.wonderland.365665.12
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848192.wonderland.365665.13
│           └── loss_validation loss
│               └── events.out.tfevents.1712848192.wonderland.365665.14
├── 512_50
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848290.wonderland.365665.15
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848292.wonderland.365665.16
│           └── loss_validation loss
│               └── events.out.tfevents.1712848292.wonderland.365665.17
├── 512_60
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848411.wonderland.365665.18
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848413.wonderland.365665.19
│           └── loss_validation loss
│               └── events.out.tfevents.1712848413.wonderland.365665.20
├── 512_70
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848552.wonderland.365665.21
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848554.wonderland.365665.22
│           └── loss_validation loss
│               └── events.out.tfevents.1712848554.wonderland.365665.23
├── 512_80
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848714.wonderland.365665.24
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848717.wonderland.365665.25
│           └── loss_validation loss
│               └── events.out.tfevents.1712848717.wonderland.365665.26
├── 512_90
│   ├── best_model
│   │   ├── ckpt_0
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   ├── ckpt_3
│   │   ├── ckpt_4
│   │   ├── ckpt_5
│   │   ├── ckpt_6
│   │   ├── ckpt_7
│   │   ├── ckpt_8
│   │   ├── ckpt_9
│   │   └── value.npy
│   ├── checkpoint
│   │   └── ckpt.pt
│   ├── log
│   │   └── v_0
│   │       ├── best_model.txt
│   │       ├── configuration.txt
│   │       ├── learning_rate.csv
│   │       ├── model.txt
│   │       ├── predictions.csv
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── validation.csv
│   └── writer
│       └── v_0
│           ├── events.out.tfevents.1712848901.wonderland.365665.27
│           ├── loss_training loss
│           │   └── events.out.tfevents.1712848905.wonderland.365665.28
│           └── loss_validation loss
│               └── events.out.tfevents.1712848905.wonderland.365665.29
└── configuration.yaml
```