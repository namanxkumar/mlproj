import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # initilize the weight matrices

        # input gate
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # forget gate
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # cell state
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # output gate
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        batch_size, seq_len, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = torch.zeros(batch_size, self.hidden_size).to(
                x.device), torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(torch.mm(x_t, self.U_i) +
                                torch.mm(h_t, self.V_i) + self.b_i)
            f_t = torch.sigmoid(torch.mm(x_t, self.U_f) +
                                torch.mm(h_t, self.V_f) + self.b_f)
            o_t = torch.sigmoid(torch.mm(x_t, self.U_o) +
                                torch.mm(h_t, self.V_o) + self.b_o)
            g_t = torch.tanh(torch.mm(x_t, self.U_c) +
                             torch.mm(h_t, self.V_c) + self.b_c)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
