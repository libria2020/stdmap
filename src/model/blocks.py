import torch
from torch import nn


class BasicBlock01(torch.nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, activation='ReLU', conv_layers=3, dropout_rate=0.0):
        super(BasicBlock01, self).__init__()

        """
           Conv -> ReLU -> ... -> Conv -> ReLU -> MaxPool
           \_______________   _______________/
                            v
                         3 times (default)
        """

        pooling = 'MaxPool1d'

        blocks = [
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        ]
        if batch_norm:
            blocks.append(nn.BatchNorm1d(out_channels))
        # Activation Layer
        blocks.append(getattr(nn, activation)())

        for i in range(1, conv_layers):
            # Conv Layer
            blocks.append(
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            # Batch Norm Layer
            if batch_norm:
                blocks.append(nn.BatchNorm1d(out_channels))
            # Activation Layer
            blocks.append(getattr(nn, activation)())

        # Pooling Layer
        blocks.append(getattr(nn, pooling)(2))

        if dropout_rate != 0.0:
            blocks.append(nn.Dropout(p=dropout_rate))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x
