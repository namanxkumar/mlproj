from http.client import ImproperConnectionState
import numpy as np
import torch
import torch.nn as nn

# > - denotes start of decoder input
# < - denotes end of decoder output
# _ - filler character

# data prep functions
def create_embedding():
    char_arr = [i for i in '><_abcdefghijklmnopqrstuvwxyz']
    embedding = {n: i for i, n in enumerate(char_arr)}
    return embedding, len(embedding)

def filler(seq_length, word):
    return word + '_'*(seq_length-len(word))

def make_train_batch(data, seq_length, embedding, n_class):
    input_batch, output_batch, target_batch = [], [], []
    for seq in data:
        seq = [filler(seq_length, i) for i in seq]
        input_w = [embedding[n] for n in seq[0]]
        output_w = [embedding[n] for n in ('>'+seq[1])]
        target_w = [embedding[n] for n in (seq[1]+'<')]

        input_batch.append(np.eye(n_class)[input_w])
        output_batch.append(np.eye(n_class)[output_w])
        target_batch.append(target_w)

    return torch.tensor(input_batch, dtype = torch.float32), torch.tensor(output_batch, dtype = torch.float32), torch.tensor(target_batch, dtype = torch.int64)

def make_test_batch(input_w, seq_length, embedding, n_class):
    input_batch, output_batch = [], []

    input_w = filler(seq_length, input_w)
    input = [embedding[n] for n in input_w]
    output = [embedding[n] for n in ('>' + '_'*seq_length)]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    return torch.tensor(input_batch, dtype = torch.float32).unsqueeze(0), torch.tensor(output_batch, dtype = torch.float32).unsqueeze(0)

def training_data():
    embedding, n_class = create_embedding()
    data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
    batch_size = len(data)
    seq_length = 5
    return (seq_length, batch_size, n_class, embedding) + make_train_batch(data, seq_length, embedding, n_class)

# Model
class Seq2Seq(nn.Module):
    def __init__(self, n_class, n_hidden, p):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=n_class,
                              hidden_size=n_hidden, batch_first = True, dropout=p)
        self.decoder = nn.RNN(input_size=n_class,
                              hidden_size=n_hidden, batch_first = True, dropout=p)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # Encoder input shape: [batch_size, sequence_length, input_size (=num_features)]
        # Encoder states shape: [batch_size, D (=1) * layers (=1), hidden_size] => [batch_size, 1, hidden_size]
        _, states = self.encoder(enc_input, enc_hidden)

        # Decoder outputs shape: [batch_size, sequence_length, D (=1) * hidden_size] => [batch_size, sequence_length, hidden_size]
        outputs, _ = self.decoder(dec_input, states)

        # Final output shape: [batch_size, sequence_length, num_features]
        model = self.fc(outputs)

        return model

if __name__ == '__main__':
    seq_length, batch_size, n_class, embedding, input_batch, output_batch, target_batch = training_data()

    learning_rate = 0.001
    epochs = 5000
    hidden_size = 128

    model = Seq2Seq(n_class, hidden_size, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        hidden = torch.zeros(1, batch_size, hidden_size)
        
        optimizer.zero_grad()

        output = model(input_batch, hidden, output_batch)

        loss = 0

        for i in range(len(target_batch)):
            # here loss takes inputs of shape [minibatch (=sequence_length), num_features (=n_class)] and [num_features (=n_class)]
            loss += criterion(output[i], target_batch[i])
        
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        
        loss.backward() # generates derivates
        optimizer.step() # adjusts parameters
    
    def translate(word):
        input_batch, output_batch = make_test_batch(word, seq_length, embedding, n_class)

        hidden = torch.zeros(1, 1, hidden_size)
        output = model(input_batch, hidden, output_batch)

        #shape of predict: [1, sequence_length, 1]
        predict = output.data.max(2, keepdim = True)
        predict = predict.indices.squeeze()
        decoded = ['><_abcdefghijklmnopqrstuvwxyz'[i.item()] for i in predict]
        end = decoded.index('<')
        translated = ''.join(decoded[:end])
        return translated.replace('_', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))