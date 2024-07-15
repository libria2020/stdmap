import numpy

from src.model.blocks import *


class MLP(nn.Module):
    def __init__(self, configuration):
        super(MLP, self).__init__()

        number_of_layers = configuration.model.num_of_layers
        neurons_per_layer = configuration.model.neurons_per_layer
        activation = configuration.model.activation
        dropout_rates = configuration.model.dropout_rates

        layers = []

        input_size = neurons_per_layer[0]

        for i in range(1, number_of_layers - 1):
            output_size = neurons_per_layer[i]
            layers.append(nn.Linear(input_size, output_size))
            layers.append(getattr(nn, activation)())
            if dropout_rates[i - 1] > 0:
                layers.append(nn.Dropout(p=dropout_rates[i - 1]))
            input_size = output_size

        layers.append(nn.Linear(input_size, neurons_per_layer[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.view(batch_size, seq_len * features)
        return self.model(x)


conv_blocks = {"01": BasicBlock01}


class Con1DModel(nn.Module):
    def __init__(self, configuration):
        super(Con1DModel, self).__init__()

        # Feature Extractor
        features_extractor = [conv_blocks[configuration.model.block](in_channels=configuration.model.in_channels,
                                                                     out_channels=configuration.model.out_channels[0],
                                                                     batch_norm=configuration.model.batch_norm,
                                                                     activation=configuration.model.activation,
                                                                     conv_layers=configuration.model.conv_layers,
                                                                     dropout_rate=configuration.model.dropout_rate)]

        for i in range(1, len(configuration.model.out_channels)):
            features_extractor.append(
                conv_blocks[configuration.model.block](in_channels=configuration.model.out_channels[i - 1],
                                                       out_channels=configuration.model.out_channels[i],
                                                       batch_norm=configuration.model.batch_norm,
                                                       activation=configuration.model.activation,
                                                       conv_layers=configuration.model.conv_layers,
                                                       dropout_rate=configuration.model.dropout_rate))

        # Global average pooling
        global_pooling = "AdaptiveAvgPool1d"
        features_extractor.append(getattr(nn, global_pooling)(1))
        latent_dimension = configuration.model.out_channels[-1]

        # number_blocks = len(features_extractor)
        # sequence_length = configuration.dataset.sequence_length
        # latent_dimension = int(sequence_length**2 / (2 ** number_blocks))

        # Classifier
        classifier = [
            nn.Linear(latent_dimension, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ]

        self.features_extractor = nn.ModuleList(features_extractor)
        self.classifier = nn.ModuleList(classifier)

    def forward(self, x):
        # Input shape: (batch_size, features, sequence_length)

        for layer in self.features_extractor:
            x = layer(x)

        # Flatten
        x = x.view(x.size(0), -1)

        for layer in self.classifier:
            x = layer(x)

        return x


class LSTMModel(nn.Module):
    def __init__(self, configuration):
        super(LSTMModel, self).__init__()

        self.input_size = configuration.model.input_size
        self.hidden_size = configuration.model.hidden_size
        self.num_layers = configuration.model.num_layers
        self.batch_first = configuration.model.batch_first
        self.dropout = configuration.model.dropout
        self.bidirectional = configuration.model.bidirectional

        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # Input shape: (batch_size, features, sequence_length) - > (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


"""
Timeseries classification with a Transformer model
	https://keras.io/examples/timeseries/timeseries_classification_transformer/

Transformer-Based Time Series with PyTorch (10.3)
	https://www.youtube.com/watch?app=desktop&v=NGzQpphf_Vc

T81-558: Applications of Deep Neural Network
Part 10.3: Transformer-Based Time Series with PyTorch
	https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_10_3_transformer_timeseries.ipynb
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # creates a tensor of zeros with dimensions `(max_len, d_model)`
        # `max_len` represents the maximum sequence length and `d_model` represents the dimensionality of the model
        # (also known as the embedding dimension).
        pe = torch.zeros(max_len, d_model)

        # creates a tensor containing values from 0 to `max_len - 1` and converts it to a float tensor.
        # then, it unsqueezes the tensor along dimension 1 to obtain a column vector.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # generates a tensor containing the exponential term for positional encoding calculation.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-numpy.log(10000.0) / d_model))
        # computes the sine of even indices of the positional encoding tensor
        pe[:, 0::2] = torch.sin(position * div_term)
        #  computes the cosine of odd indices of the positional encoding tensor
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshapes the positional encoding tensor by unsqueezing it along dimension 0
        # this reshaping operation makes the tensor compatible with batched input
        # expected input shape is (batch_size, sequence_length, features)
        pe = pe.unsqueeze(0)

        # registers the positional encoding tensor as a buffer of the `PositionalEncoding` module
        # buffers are parameters that are not updated during optimization but are used by the module during forward computation.
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, configuration):
        super(TransformerModel, self).__init__()

        self.input_dim = configuration.model.input_dim
        self.d_model = configuration.model.d_model
        self.nhead = configuration.model.nhead
        self.num_layers = configuration.model.num_layers
        self.dropout = configuration.model.dropout
        self.out_features = configuration.model.out_features
        self.max_len = configuration.model.max_len

        self.encoder = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.max_len)

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)

        self.decoder = nn.Linear(self.d_model, self.out_features)

    def forward(self, x):
        # Input shape: (batch_size, features, sequence_length) - > (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])

        return x


# if __name__ == '__main__':
#     path = Path('../../configuration/configuration.yaml')
#
#     with path.open('r') as f:
#         configuration = edict(yaml.safe_load(f))
#
#     # Input shape: (batch_size, features, sequence_length)
#     x = torch.randn((32, 2, 512))
#     model = TransformerModel(configuration)
#
#     print(model(x).shape)

# TODO: add

"""
input size: (batch_size, features, sequence_length)

Model:
    LSTM
    
Model: 
    Transformer

Model:
    Basic Block -> ... > Basic Block -> Global Pooling -> Dense 
    \_____________   _____________/
                   v
                x times
                  
Model:
    Basic Block -> ... > Basic Block -> Flatten -> Dense [1]
    \_____________   _____________/
                   v
                x times
                
                        
Basic Block [1]

    Conv1d ->  BatchNorm1d (optional) -> Activation -> Pooling
    \_____________________   _____________________/
                           v
                        x times

[1] Corbetta, A., Menkovski, V., Benzi, R., & Toschi, F. (2021). 
Deep learning velocity signals allow quantifying turbulence intensity. Science Advances, 7(12), eaba7281.

            
output size: 1 
"""
