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
        print(x)
        x = self.classifier(x[:,-1]) # classify based on last output of the encoder
        x = torch.sigmoid(x)
        return x

BATCH_SIZE = 10
ITERATIONS = 1000
SEQ_LENGTH = 50 # 441344
LSTM_SIZE = 64

class LstmAutoencoder(nn.Module):
    def __init__(self, device):
        super(LstmAutoencoder, self).__init__()
        self.device = device
        self.encoder = nn.LSTM(input_size=1, hidden_size=LSTM_SIZE)
        self.decoder = nn.LSTM(input_size=1, hidden_size=LSTM_SIZE)
        self.linear = nn.Linear(LSTM_SIZE, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        _, last_state = self.encoder(x)
        outs_total = torch.zeros(SEQ_LENGTH, BATCH_SIZE, 1, device=self.device)
        decoder_input = torch.zeros(1, BATCH_SIZE, 1, device=self.device)
        for i in range(SEQ_LENGTH):
            outs, last_state = self.decoder(decoder_input, last_state)
            outs = self.linear(outs)
            outs = self.softmax(outs)
            # outs.squeeze_(2)
            outs_total[i,...] = outs
        return outs_total
