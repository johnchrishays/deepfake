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
    def __init__(self, n_vid_features, n_aud_features, n_head, n_layers, n_linear_hidden=30, dropout=0.3):
        super(Classifier, self).__init__()
        vid_encoder_layer = nn.TransformerEncoderLayer(d_model=n_vid_features, nhead=n_head)
        self.vid_transformer_encoder = nn.TransformerEncoder(vid_encoder_layer, num_layers=n_layers)
        aud_encoder_layer = nn.TransformerEncoderLayer(d_model=n_aud_features, nhead=1)
        self.aud_transformer_encoder = nn.TransformerEncoder(aud_encoder_layer, num_layers=n_layers)
        self.dense = nn.Linear(n_vid_features + n_aud_features, n_linear_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.out_pred = nn.Linear(n_linear_hidden, 1)

    def forward(self, vid, aud):
        vid = self.vid_transformer_encoder(vid)
        aud = self.aud_transformer_encoder(aud)
        x = torch.cat((vid[:,-1], aud[:,-1]), 1) # classify based on last output of the encoder
        x = self.dropout(x)
        x = self.dense(x) 
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.out_pred(x)
        return x

class LstmAutoencoder(nn.Module):
    def __init__(self, device, batch_size, seq_length, lstm_size):
        super(LstmAutoencoder, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_size = lstm_size

        self.encoder = nn.LSTM(input_size=1, hidden_size=lstm_size, num_layers=2, dropout=0.3)
        # self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(input_size=1, hidden_size=lstm_size, num_layers=2, dropout=0.3)
        self.linear = nn.Linear(lstm_size, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        _, last_state = self.encoder(x)
        outs_total = torch.zeros(self.seq_length, self.batch_size, 1, device=self.device)
        decoder_input = torch.zeros(1, self.batch_size, 1, device=self.device)
        for i in range(self.seq_length):
            outs, last_state = self.decoder(decoder_input, last_state)
            outs = self.linear(outs)
            outs = self.softmax(outs)
            outs_total[i,...] = outs
        return outs_total
