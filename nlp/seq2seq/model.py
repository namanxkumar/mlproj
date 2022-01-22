from http.client import ImproperConnectionState
import numpy as np
import torch
import torch.nn as nn

# Model
class Seq2Seq(nn.Module):
    def __init__(self, n_class, n_hidden, p):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size = n_class, hidden_size = n_hidden, dropout = p)
        self.decoder = nn.RNN(input_size = n_class, hidden_size = n_hidden, dropout = p)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # Encoder input shape: [sequence_length, batch_size, input_size (=num_features)]
        # Encoder states shape: [D (=1) * layers (=1), batch_size, hidden_size] => [1, batch_size, hidden_size]
        _, states = self.encoder(enc_input, enc_hidden)
        
        # Decoder outputs shape: [sequence_length, batch_size, D (=1) * hidden_size] => [sequence_length, batch_size, hidden_size]
        outputs, _ = self.decoder(dec_input, states)

        # Final output shape: [sequence_length, batch_size, num_features]
        model = self.fc(outputs)

        return model