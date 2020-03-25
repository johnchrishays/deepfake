import torch.nn as nn

# TODO: possibly look into multiple filter sizes
N_IN_CHANNELS = 3

class Autoencoder(nn.Module):
    def __init__(self, n_out_channels1=10, n_out_channels2=10, n_out_channels3=6, kernel_size=5):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(N_IN_CHANNELS, n_out_channels1, kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n_out_channels1, n_out_channels2, kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(n_out_channels2, n_out_channels3, kernel_size, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(n_out_channels3, n_out_channels2, kernel_size, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(n_out_channels2, n_out_channels1, kernel_size, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(n_out_channels1, N_IN_CHANNELS, kernel_size, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

