import torch
import torch.nn as nn
import torch.nn.functional as F


class ABCNN(nn.Module):
    def __init__(self, device, batch_size, vocab_size, embed_dim, kernel_width, sentence1_length=15, sentence2_length=15):
        super(ABCNN, self).__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.sentence1_length = sentence1_length
        self.sentence2_length = sentence2_length

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_width, 1), padding=(2, 0))
        self.conv2d2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_width, 1), padding=(2, 0))
        self.W0 = nn.Parameter(torch.randn((sentence1_length, embed_dim), requires_grad=True))
        self.W1 = nn.Parameter(torch.randn((sentence2_length, embed_dim), requires_grad=True))
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(2, kernel_width, 1), padding=(0, 2, 0))
        self.Attention1 = Attention(device)
        self.Attention2 = Attention(device)
        self.dnn = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, sentence1, sentence2):
        rep_s1 = F.avg_pool2d(self.conv2d1(self.embed(sentence1).reshape(len(sentence1), 1, self.sentence1_length, self.embed_dim)
                                           ).reshape(len(sentence1), self.sentence1_length + 2, self.embed_dim), kernel_size=(3, 1), stride=1)
        rep_s2 = F.avg_pool2d(self.conv2d1(self.embed(sentence2).reshape(len(sentence2), 1, self.sentence2_length, self.embed_dim)
                                           ).reshape(len(sentence2), self.sentence2_length + 2, self.embed_dim), kernel_size=(3, 1), stride=1)
        A1 = self.Attention1.attention(len(sentence1), rep_s1, rep_s2)
        att_s1 = torch.matmul(A1, self.W0)
        att_s2 = torch.matmul(A1, self.W1)
        conv_input_s1 = torch.cat((rep_s1.reshape(len(sentence1), 1, 1, self.sentence1_length, self.embed_dim),
                                att_s1.reshape(len(sentence1), 1, 1, self.sentence1_length, self.embed_dim)), dim=2)
        conv_input_s2 = torch.cat((rep_s2.reshape(len(sentence2), 1, 1, self.sentence2_length, self.embed_dim),
                                   att_s2.reshape(len(sentence2), 1, 1, self.sentence2_length, self.embed_dim)), dim=2)
        conv2_s1 = self.conv3d(conv_input_s1).reshape(len(sentence1), self.sentence1_length + 2, self.embed_dim)
        conv2_s2 = self.conv3d(conv_input_s2).reshape(len(sentence2), self.sentence2_length + 2, self.embed_dim)
        A2 = self.Attention2.attention(len(sentence2), conv2_s1, conv2_s2)
        col_wise_sum = torch.sum(A2, dim=2)
        row_wise_sum = torch.sum(A2, dim=1)
        conv2_att_s1 = F.avg_pool2d(torch.mul(conv2_s1, col_wise_sum.unsqueeze(2)), kernel_size=(3, 1), stride=1)
        conv2_att_s2 = F.avg_pool2d(torch.mul(conv2_s2, row_wise_sum.unsqueeze(2)), kernel_size=(3, 1), stride=1)
        vec_s1 = F.avg_pool2d(self.conv2d2(conv2_att_s1.reshape(len(sentence1), 1, self.sentence1_length, self.embed_dim)
                                         ).reshape(len(sentence1), self.sentence1_length + 2, self.embed_dim),
                              kernel_size=(self.sentence1_length+2, 1), stride=1).squeeze()
        vec_s2 = F.avg_pool2d(self.conv2d2(conv2_att_s2.reshape(len(sentence2), 1, self.sentence2_length, self.embed_dim)
                                         ).reshape(len(sentence2), self.sentence2_length + 2, self.embed_dim),
                              kernel_size=(self.sentence2_length+2, 1), stride=1).squeeze()
        output = self.dnn(torch.cat((vec_s1, vec_s2), dim=1))
        return output


class Attention(nn.Module):
    def __init__(self, device):
        super(Attention, self).__init__()
        self.device =device

    def attention(self, batch_size, sentence1, sentence2):
        A1 = [[[0]*len(sentence2[0])]*len(sentence1[0])]*batch_size
        for k in range(batch_size):
            for i in range(len(sentence1[0])):
                for j in range(len(sentence2[0])):
                    tmp = sentence1[k][i] - sentence2[k][j]
                    tmp = tmp * tmp
                    score = 1/(1+tmp.sum().sqrt())
                    A1[k][i][j] = score
        return torch.Tensor(A1).to(self.device)

