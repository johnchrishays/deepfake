import torch
import torch.nn as nn
import math

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

class FaceAutoencoder(nn.Module):
    def __init__(self, n_out_channels1=4, n_out_channels2=4, n_out_channels3=1, \
                kernel_size1=5, kernel_size2=5, kernel_size3=5):
        super(FaceAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(N_IN_CHANNELS, out_channels=n_out_channels1, kernel_size=kernel_size1, stride=2, padding=2), # [3,160,160] -> [4, 80, 80]
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Conv2d(in_channels=n_out_channels1, out_channels=n_out_channels2, kernel_size=kernel_size2, stride=2, padding=2), # [3,80,80] -> [4, 40, 40]
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=0), # [4, 40, 40] -> [4, 38, 38]

            nn.Dropout(0.3),

            nn.Conv2d(in_channels=n_out_channels2, out_channels=n_out_channels3, kernel_size=kernel_size3, stride=1, padding=1), # [4, 36, 36] -> [1, 36, 36]
            nn.MaxPool2d(5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(n_out_channels3, n_out_channels2, kernel_size1, stride=1, padding=2), # [1, 36, 36] -> [4, 36, 36]
            nn.ReLU(),
            # nn.MaxPool2d(5, stride=3, padding=2),
            nn.Upsample(size=(40, 40)),

            nn.Dropout(0.3),

            nn.Conv2d(n_out_channels2, n_out_channels1, kernel_size2, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Dropout(0.3),

            nn.Conv2d(n_out_channels1, N_IN_CHANNELS, kernel_size3, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        # print(x.size())
        x = self.decoder(x)
        # print(x.size())
        return x

    def encode(self, x):
        return self.encoder(x)


# copied from 
# https://github.com/pytorch/examples/tree/master/word_language_model
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Classifier(nn.Module):
    def __init__(self, n_vid_features, n_aud_features, n_head, n_layers, n_linear_hidden=30, dropout=0.1):
        super(Classifier, self).__init__()
        # video
        self.vid_pos_encoder = PositionalEncoding(d_model=n_vid_features)
        vid_encoder_layer = nn.TransformerEncoderLayer(d_model=n_vid_features, nhead=n_head)
        self.vid_transformer_encoder = nn.TransformerEncoder(vid_encoder_layer, num_layers=n_layers)
        #self.dropout = nn.Dropout(p=dropout)
        self.vid_pred = nn.Linear(n_vid_features, 1)

        # audio
        self.aud_pos_encoder = PositionalEncoding(d_model=n_aud_features)
        aud_encoder_layer = nn.TransformerEncoderLayer(d_model=n_aud_features, nhead=1)
        self.aud_transformer_encoder = nn.TransformerEncoder(aud_encoder_layer, num_layers=n_layers)
        # combine video and audio
        self.out_pred = nn.Linear(2, 1)

    def forward(self, vid, aud):
        vid = vid.permute(1, 0, 2)
        vid = self.vid_pos_encoder(vid)
        vid = self.vid_transformer_encoder(vid)
        #vid = self.dropout(vid)
        vid = self.vid_pred(vid) 
        vid = torch.sigmoid(vid)
        vid = torch.mean(vid, axis=0)
        #vid = self.dropout(vid)
        # print("video size:", vid.size())
        #vid_pred = self.out_vid_pred(vid[-1])

        aud = aud.permute(1, 0, 2)
        aud = self.aud_pos_encoder(aud)
        aud = self.aud_transformer_encoder(aud)
        aud = torch.sigmoid(aud)
        aud = torch.mean(aud, axis=0)
        # print("audio size:", aud.size())
        x = torch.cat((vid, aud), 1) # classify based on last output of the encoder
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
