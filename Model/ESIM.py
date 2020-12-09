import torch
import torch.nn as nn


class ESIM(nn.Module):
    def __init__(self, mode, device, batch_size, vocab_size, sentence_length, embed_dim=100, hidden_size=256):
        super(ESIM,self).__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.h0_lstm1 = torch.zeros([batch_size, 1, hidden_size * 2]).to(device)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if self.mode == 'bi':
            self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True).to(device)
            self.lstm2 = nn.LSTM(input_size=hidden_size * 6, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True).to(device)
        elif self.mode == 'tree':
            pass
        else:
            Warning('mode 必须为 bi 或 tree ！')
        self.matrix = LocalInference(device).to(device)
        self.dense = nn.Linear(hidden_size * 8, hidden_size * 4).to(device)
        self.pooling = Pooling().to(device)
        self.dnn = nn.Sequential(
            nn.Linear(self.hidden_size * 8, hidden_size),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2),
            nn.Softmax(dim=0)
        ).to(device)

    def forward(self, a, b):
        embed_a = self.embed(a)
        a, _ = self.lstm1(embed_a)
        embed_b = self.embed(b)
        b, _ = self.lstm1(embed_b)
        self.matrix.matrix(a, b)
        self.matrix.local_relevance()
        self.matrix.enhancement()
        f_a = self.dense(self.matrix.ma).relu()
        f_b = self.dense(self.matrix.mb).relu()
        h_n_a = torch.cat((self.h0_lstm1[:len(a)], a), dim=1)
        h_n_b = torch.cat((self.h0_lstm1[:len(a)], b), dim=1)
        va_input = torch.cat((f_a, h_n_a.permute(1, 0, 2)[:-1].permute(1, 0, 2)), dim=2)
        vb_input = torch.cat((f_b, h_n_b.permute(1, 0, 2)[:-1].permute(1, 0, 2)), dim=2)
        va, _ = self.lstm2(va_input)
        vb, _ = self.lstm2(vb_input)
        vector = self.pooling.final_vector(va, vb)
        output = self.dnn(vector)
        return output


class LocalInference(nn.Module):
    def __init__(self, device):
        super(LocalInference, self).__init__()
        self.device = device

    def matrix(self, a, b):
        self.a = a
        self.b = b
        self.matrix_ = self.a @ self.b.permute(0, 2, 1)

    def local_relevance(self):
        weight_a = self.matrix_.softmax(dim=2)
        weight_b = self.matrix_.softmax(dim=1)
        self.a_ = weight_a @ self.b
        self.b_ = weight_b @ self.a

    def enhancement(self):
        self.ma = torch.cat((self.a, self.a_, self.a - self.a_, self.a * self.a_), dim=2)
        self.mb = torch.cat((self.b, self.b_, self.b - self.b_, self.b * self.b_), dim=2)


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def final_vector(self, va, vb):
        va_avg = torch.sum(va, dim=1)
        va_max = torch.max(va, dim=1)[0]
        vb_avg = torch.sum(vb, dim=1)
        vb_max = torch.max(vb, dim=1)[0]
        vector = torch.cat((va_avg, va_max, vb_avg, vb_max), dim=1)
        return vector
