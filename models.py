import torch.nn as nn

# TODO: possibly look into multiple filter sizes
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 8, 5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 16, 5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 3, 5, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

