import torch
from torch import nn


class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers, dropout, device, fea_type='embed',
                 rnn_type='RNN'):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # hidden_size
        self.num_classes = num_classes
        self.fea_type = fea_type
        self.rnn_type = rnn_type
        if self.fea_type == 'embed':
            self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        else:
            self.word_to_vec = self.to_onehot
        if rnn_type == 'RNN':
            print('using RNN')
            self.rnn = nn.RNN(num_classes, hidden_size, num_layers)
        elif rnn_type == 'LSTM':
            print('using LSTM')
            self.rnn = nn.LSTM(num_classes, hidden_size, num_layers, dropout=dropout, bias=True, batch_first=False)
        else:
            print('using GRU')
            self.rnn = nn.GRU(num_classes, hidden_size, num_layers, dropout=dropout, bias=True, batch_first=False)
        self.project = nn.Linear(hidden_size, num_classes)
        self.device = device

    def one_hot(self, x, n_class, dtype=torch.float32):
        x = x.long()
        res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
        res.scatter_(1, x.view(-1, 1), 1)
        return res

    def to_onehot(self, X):
        return [self.one_hot(X[:, i], self.num_classes) for i in range(X.shape[1])]

    def forward(self, x, hs=None):
        # pdb.set_trace()
        batch = x.shape[0]
        if hs is None:
            if self.rnn_type == 'RNN':
                hs = torch.zeros(self.num_layers, batch, self.hidden_size).to(self.device)
            elif self.rnn_type == 'LSTM':
                hs = (torch.zeros(self.num_layers, batch, self.hidden_size).to(self.device),
                      torch.zeros(self.num_layers, batch, self.hidden_size).to(self.device))

        if self.fea_type == 'embed':
            word_embed = self.word_to_vec(x)  # (batch, len, embed) [128, 20, 512]
            word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed) [20, 128, 512]
            out, h0 = self.rnn(word_embed, hs)

        else:
            word_embed = self.to_onehot(x)
            out, h0 = self.rnn(torch.stack(word_embed), hs)  # out: [20, 128, 512]  h0: [2, 128, 512]

        le, mb, hd = out.shape
        out = out.view(le * mb, hd)  # [2560, 512]
        out = self.project(out)  # [2560, 2581]
        out = out.view(le, mb, -1)  # [20, 128, 2581]
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden) [128, 20, 2581]
        return out.view(-1, out.shape[2]), h0  # [2560, 2581], [2, 128, 512]
