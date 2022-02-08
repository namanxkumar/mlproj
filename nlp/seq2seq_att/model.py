from unicodedata import bidirectional
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout):
        super(EncoderRNN, self).__init__()
        '''
            input_size: Length of sentence with start and end tokens
            embedding_size: Size of embedding
            hidden_size: RNN Hidden vector size
            dropout: Dropout rate
        '''

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, sentences):
        # sentences = [batch_size, input_size]

        embedded = self.dropout(self.embedding(sentences))

        # embedded = [batch_size, input_size, embedding_size]

        outputs, hidden = self.rnn(embedded) # we do not provide initial hidden states, and they are initiliazed to zero

        # outputs = [batch_size, input_size, 2*hidden_size] -> output for each word in sentence across both forward and backward direction
        # hidden = [2*1, batch_size, hidden_size]  -> last hidden state for both directions, for every layer (=1)

        # now to get the feed hidden state for decoder, we must concatenate the forward and backward hidden states and pass them
        # through a fully connected layer to change the shape

        # hidden[-2, :, :] and hidden[-1, :, :] fetch the last layer hidden states for the forward and backward pass respectively

        hidden = torch.relu(self.fc(torch.cat(hidden[-2, :, :], hidden[-1, :, :])))

        # outputs = [batch_size, input_size, 2*hidden_size]
        # hidden = [batch_size, hidden_size]

        return outputs, hidden

