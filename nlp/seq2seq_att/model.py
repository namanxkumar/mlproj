from json import encoder
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout):
        super(Encoder, self).__init__()
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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attention = nn.Linear(hidden_size*3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):

        # hidden = [batch_size, hidden_size]
        # encoder_outputs = [batch_size, input_size, 2*hidden_size]

        input_size = encoder_outputs.shape[1]

        # concatenate previous hidden state to every encoder output
        hidden = hidden.unsqueeze(1).repeat(1, input_size, 1)

        # hidden shape after unsqueeze -> [batch_size, 1, hidden_size]
        # hidden (finally) = [batch_size, input_size, hidden_size]

        energy = torch.relu(self.attention(torch.cat((hidden, encoder_outputs), dim = 2)))

        # energy = [batch_size, input_size, hidden_size]

        attention: torch.Tensor = self.v(energy)
        
        # attention = [batch_size, input_size, 1]

        attention = attention.squeeze(2)

        # attention = [batch_size, input_size]

        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, dropout, attention):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU((hidden_size*2) + embedding_size, hidden_size, batch_first = True)
        self.fc = nn.Linear((hidden_size*3) + embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size]
        # hidden = [batch_size, hidden_size]
        # encoder_outputs = [batch_size, input_size, 2*hidden_size]

        # rough flow: embedding + (attention*encoder_outputs) + prev_hidden_state -> rnn -> hidden_state + (attention*encoder_outputs) + embedding -> fc -> output

        input = input.unsqueeze(1)

        # input = [batch_size, 1]

        embedded = self.dropout(self.embedding(input))

        # embedded = [batch_size, 1, embedding_size]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch_size, input_size]

        a = a.unsqueeze(1)

        # a = [batch_size, 1, input_size]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch_size, 1, 2*hidden_size]

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        # rnn_input = [batch_size, 1, (hidden_size*2) + embedding_size]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [batch_size, 1, hidden_size]
        # hidden = [1, batch_size, hidden_size]
        assert (output == hidden.permute(1, 0, 2)).all()

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))

        # prediction = [batch_size, output_size]

        return prediction, hidden.squeeze(0)
        
