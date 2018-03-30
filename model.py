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
    

class DualMcepNet(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, bidirectional):
        super(DualMcepNet, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim

        self.pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim * self.num_direction, out_dim)
        )

        self.dual_pre_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU()
        )
        self.dual_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)

        self.dual_post_net = nn.Sequential(
            nn.Linear(hidden_dim * self.num_direction, out_dim)
        )

    def forward(self, x, lengths, h, c, dual=False):
        if not dual:
            output = self.normal_forward(x, lengths, h, c)
            dual_output = self.dual_forward(output, lengths, h, c)
            return output, dual_output
        else:
            dual_output = self.dual_forward(x, lengths, h, c)
            output = self.normal_forward(dual_output, lengths, h, c)
            return output, dual_output

    def normal_forward(self, x, lengths, h, c):
        output = self.pre_net(x)
        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(output, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.post_net(output)
        return output

    def dual_forward(self, x, lengths, h, c):
        dual_output = self.dual_pre_net(x)
        dual_output = torch.nn.utils.rnn.pack_padded_sequence(dual_output, lengths, batch_first=True)
        self.dual_lstm.flatten_parameters()
        dual_output, _ = self.dual_lstm(dual_output, (h, c))
        dual_output, _ = torch.nn.utils.rnn.pad_packed_sequence(dual_output, batch_first=True)
        dual_output = self.dual_post_net(dual_output)
        return dual_output

    def init_hidden(self, batch_size):
        h, c = Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim)),\
               Variable(torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_dim))
        return h, c





