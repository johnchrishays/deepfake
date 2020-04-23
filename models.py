import torch
import torch.nn as nn

# TODO: possibly look into multiple filter sizes
N_IN_CHANNELS = 3



class Autoencoder(nn.Module):
    def __init__(self, n_out_channels1=4, n_out_channels2=4, n_out_channels3=1, \
                kernel_size1=5, kernel_size2=5, kernel_size3=5):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(N_IN_CHANNELS, out_channels=n_out_channels1, kernel_size=kernel_size1, stride=2, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(4, stride=2, padding=1),

            nn.Conv2d(in_channels=n_out_channels1, out_channels=n_out_channels2, kernel_size=kernel_size2, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=1, padding=2),

            nn.Conv2d(in_channels=n_out_channels2, out_channels=n_out_channels3, kernel_size=kernel_size3, stride=3, padding=2),
            nn.MaxPool2d(5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=3),
            nn.Conv2d(n_out_channels3, n_out_channels2, kernel_size1, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(5, stride=3, padding=2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(n_out_channels2, n_out_channels1, kernel_size2, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(n_out_channels1, N_IN_CHANNELS, kernel_size3, stride=1, padding=2),
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.size())
        x = self.decoder(x)
        # print(x.size())
        return x

    def encode(self, x):
        return self.encoder(x)

class Classifier(nn.Module):
    def __init__(self, n_features, n_head, n_layers):
        super(Classifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(n_features, 1)
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.classifier(x[-1]) # classify based on last output of the encoder
        x = torch.sigmoid(x)
        return x
