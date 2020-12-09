import torch
import torch.nn as nn


class DRCN(nn.Module):
    def __init__(self,device, batch_size, vocab_size, embed_dim, hidden_size=100):
        super(DRCN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstmcell1 = nn.LSTMCell(embed_dim, hidden_size)    # 修改为四种embedding的格式
        self.lstmcell2 = nn.LSTMCell(hidden_size * 5, hidden_size)
        self.lstmcell3 = nn.LSTMCell(hidden_size * 7, hidden_size)
        self.lstmcell4 = nn.LSTMCell(hidden_size * 9, hidden_size)
        self.lstmcell5 = nn.LSTMCell(hidden_size * 11, hidden_size)
        self.c0 = torch.ones(batch_size, hidden_size).to(device)
        self.h0 = torch.ones(batch_size, hidden_size).to(device)
        self.attention = Attention(device)
        self.autoencoder1 = nn.Sequential(
            nn.Linear(hidden_size * 9, 200),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(200, hidden_size * 9),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.autoencoder2 = nn.Sequential(
            nn.Linear(hidden_size * 13, 200),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(200, hidden_size * 13),
            nn.Dropout(0.2),
            nn.Tanh()
        )
        self.dnn = nn.Sequential(
            nn.Linear(hidden_size * 13 * 5, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, p, q):
        x_p_1 = self.embed(p).permute(1, 0, 2)
        x_q_1 = self.embed(q).permute(1, 0, 2)
        # 第一层
        hx = self.h0
        cx = self.c0
        h_p_1 = None
        for i in range(len(x_p_1)):
            hx, cx = self.lstmcell1(x_p_1[0], (hx, cx))
            if i == 0:
                h_p_1 = hx.unsqueeze(0)
            else:
                h_p_1 = torch.cat((h_p_1, hx.unsqueeze(0)), dim=0)
        h_q_1 = None
        for i in range(len(x_q_1)):
            hx, cx = self.lstmcell1(x_q_1[0], (hx, cx))
            if i == 0:
                h_q_1 = hx.unsqueeze(0)
            else:
                h_q_1 = torch.cat((h_q_1, hx.unsqueeze(0)), dim=0)
        # 第二层
        hx = self.h0
        cx = self.c0
        a_p_1, a_q_1 = self.attention.attention(h_p_1.permute(1, 0, 2), h_q_1.permute(1, 0 ,2))
        x_p_2 = torch.cat((h_p_1, a_p_1.permute(1, 0, 2), x_p_1), dim=2)
        x_q_2 = torch.cat((h_q_1, a_q_1.permute(1, 0, 2), x_q_1), dim=2)
        h_p_2 = None
        for i in range(len(x_p_2)):
            hx, cx = self.lstmcell2(x_p_2[0], (hx, cx))
            if i == 0:
                h_p_2 = hx.unsqueeze(0)
            else:
                h_p_2 = torch.cat((h_p_2, hx.unsqueeze(0)), dim=0)
        h_q_2 = None
        for i in range(len(x_q_2)):
            hx, cx = self.lstmcell2(x_q_2[0], (hx, cx))
            if i == 0:
                h_q_2 = hx.unsqueeze(0)
            else:
                h_q_2 = torch.cat((h_q_2, hx.unsqueeze(0)), dim=0)
        # 第三层
        hx = self.h0
        cx = self.c0
        a_p_2, a_q_2 = self.attention.attention(h_p_2.permute(1, 0, 2), h_q_2.permute(1, 0, 2))
        x_p_3 = torch.cat((h_p_2, a_p_2.permute(1, 0, 2), x_p_2), dim=2)
        x_q_3 = torch.cat((h_q_2, a_q_2.permute(1, 0, 2), x_q_2), dim=2)
        h_p_3 = None
        for i in range(len(x_p_3)):
            hx, cx = self.lstmcell3(x_p_3[0], (hx, cx))
            if i == 0:
                h_p_3 = hx.unsqueeze(0)
            else:
                h_p_3 = torch.cat((h_p_3, hx.unsqueeze(0)), dim=0)
        h_q_3 = None
        for i in range(len(x_q_3)):
            hx, cx = self.lstmcell3(x_q_3[0], (hx, cx))
            if i == 0:
                h_q_3 = hx.unsqueeze(0)
            else:
                h_q_3 = torch.cat((h_q_3, hx.unsqueeze(0)), dim=0)
        # 自编码器
        a_p_3, a_q_3 = self.attention.attention(h_p_3.permute(1, 0, 2), h_q_3.permute(1, 0, 2))
        x_p_en = torch.cat((h_p_3, a_p_3.permute(1, 0, 2), x_p_3), dim=2)
        x_q_en = torch.cat((h_q_3, a_q_3.permute(1, 0, 2), x_q_3), dim=2)
        x_p_4 = self.autoencoder1(x_p_en)
        x_q_4 = self.autoencoder1(x_q_en)
        # 第四层
        hx = self.h0
        cx = self.c0
        h_p_4 = None
        for i in range(len(x_p_4)):
            hx, cx = self.lstmcell4(x_p_4[0], (hx, cx))
            if i == 0:
                h_p_4 = hx.unsqueeze(0)
            else:
                h_p_4 = torch.cat((h_p_4, hx.unsqueeze(0)), dim=0)
        h_q_4 = None
        for i in range(len(x_q_4)):
            hx, cx = self.lstmcell4(x_q_4[0], (hx, cx))
            if i == 0:
                h_q_4 = hx.unsqueeze(0)
            else:
                h_q_4 = torch.cat((h_q_4, hx.unsqueeze(0)), dim=0)
        # 第五层
        hx = self.h0
        cx = self.c0
        a_p_4, a_q_4 = self.attention.attention(h_p_4.permute(1, 0, 2), h_q_4.permute(1, 0, 2))
        x_p_5 = torch.cat((h_p_4, a_p_4.permute(1, 0, 2), x_p_4), dim=2)
        x_q_5 = torch.cat((h_q_4, a_q_4.permute(1, 0, 2), x_q_4), dim=2)
        h_p_5 = None
        for i in range(len(x_p_5)):
            hx, cx = self.lstmcell5(x_p_5[0], (hx, cx))
            if i == 0:
                h_p_5 = hx.unsqueeze(0)
            else:
                h_p_5 = torch.cat((h_p_5, hx.unsqueeze(0)), dim=0)
        h_q_5 = None
        for i in range(len(x_q_5)):
            hx, cx = self.lstmcell5(x_q_5[0], (hx, cx))
            if i == 0:
                h_q_5 = hx.unsqueeze(0)
            else:
                h_q_5 = torch.cat((h_q_5, hx.unsqueeze(0)), dim=0)
        # 自编码器
        a_p_5, a_q_5 = self.attention.attention(h_p_5.permute(1, 0, 2), h_q_5.permute(1, 0, 2))
        x_p_en = torch.cat((h_p_5, a_p_5.permute(1, 0, 2), x_p_5), dim=2)
        x_q_en = torch.cat((h_q_5, a_q_5.permute(1, 0, 2), x_q_5), dim=2)
        x_p_pool = self.autoencoder2(x_p_en)
        x_q_pool = self.autoencoder2(x_q_en)
        # 池化层
        vec_p = x_p_pool.permute(1, 0, 2).max(dim=1)[0]
        vec_q = x_q_pool.permute(1, 0, 2).max(dim=1)[0]
        vector = torch.cat((vec_p, vec_q, vec_p + vec_q, vec_p - vec_q, (vec_p - vec_q).abs()), dim=1)
        output = self.dnn(vector)
        return output


class Attention(nn.Module):
    def __init__(self, device):
        super(Attention, self).__init__()
        self.device = device

    def attention(self, p, q):
        att = torch.Tensor([[[0] * len(q[0])] * len(p[0])] * len(q)).to(self.device)
        for k in range(len(p)):
            for i in range(len(p[k])):
                for j in range(len(q[k])):
                    att[k][i][j] = torch.cosine_similarity(p[k][i], q[k][j], dim=0)
        vec_att_p = att.softmax(dim=2) @ q
        vec_att_q = att.softmax(dim=1) @ p
        return vec_att_p, vec_att_q


class Embedding(nn.Module):
    def __init__(self):
        """
            四种embedding方法
        """
        super(Embedding, self).__init__()
