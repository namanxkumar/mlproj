from turtle import position
import torch
import torch.nn as nn
import torch.optim as optim

class MultiHeadAttention(nn.Module):
    pass

class PositionwiseFeedForward(nn.Module):
    pass

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, pf_size, dropout, device, max_length = 100):
        super().__init__()

        self.device = device

        self.token_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_length, hidden_size) # This is a learned positional embedding unlike the original paper, which used static embeddings. We set the vocab size to 100, allowing a maximum token length of 100.

        self.layers = nn.ModuleList([self.EncoderLayer(hidden_size, num_heads, pf_size, dropout, device) for i in num_layers])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device) # scaling factor is the square root of the hidden dimension

    def forward(self, source, source_mask):
        # source = [batch_size, source_length]
        # source_mask = [batch_size, 1, 1, source_length]

        batch_size = source.shape[0]
        source_length = source.shape[1]

        positions = torch.arange(source_length).repeat(batch_size, 1).to(self.device)
        # positions = [batch_size, source_length]

        source = self.dropout((self.token_embedding(source) * self.scale) + self.positional_embedding(positions))
        
        # source = [batch_size, source_length, hidden_size]

        for layer in self.layers:
            source = layer(source, source_mask)

        # source = 

        return source

    class EncoderLayer(nn.Module):
        def __init__(self, hidden_size, num_heads, pf_size, dropout, device):
            super().__init__()

            self.attn_layer_norm = nn.LayerNorm(hidden_size)
            self.ff_layer_norm = nn.LayerNorm(hidden_size)
            self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout, device)
            self.positionwise_ff = PositionwiseFeedForward(hidden_size, pf_size, dropout)

            self.dropout = nn.Dropout(dropout)

        def forward(self, source, source_mask):
            # source = [batch_size, source_length, hidden_size]
            # source_mask = [batch_size, 1, 1, source_length]

            # self-attention
            _source, _ = self.self_attention(source, source, source, source_mask)

            # dropout, residual connection and layer norm
            source = self.attn_layer_norm(source + self.dropout(_source))

            # source = [batch_size, source_length, hidden_size]

            # positionwise feed forward
            _source = self.positionwise_ff(source)

            # dropout, residual and layer norm
            source = self.ff_layer_norm(source + self.dropout(_source))

            # source = [batch_size, source_length, hidden_size]

            return source

            