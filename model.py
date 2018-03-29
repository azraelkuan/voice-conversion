import torch
import torch.nn as nn
from torch.autograd import Variable


class McepNet(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, bidirectional):
        super(McepNet, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim

        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)

        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim*self.num_direction, out_dim)
        )

    def forward(self, x, lengths, h, c):
        output = self.pre_net(x)

        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(output, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.post_net(output)
        return output

    def init_hidden(self, batch_size):
        h, c = Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),\
               Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim))
        return h, c

