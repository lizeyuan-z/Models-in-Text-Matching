import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_size_att, hidden_size_ffn, dropout, head_num):
        super(Attention, self).__init__()
        self.head_num = head_num
        self.hidden_size_att = hidden_size_att

        self.WQ = []
        self.WK = []
        self.WV = []
        for i in range(head_num):
            self.WQ.append(nn.Parameter(torch.randn(embed_dim, hidden_size_att), requires_grad=True))
            self.WK.append(nn.Parameter(torch.randn(embed_dim, hidden_size_att), requires_grad=True))
            self.WV.append(nn.Parameter(torch.randn(embed_dim, hidden_size_att), requires_grad=True))
        self.WO = nn.Parameter(torch.randn(hidden_size_att * head_num, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_size_ffn),
            nn.ReLU(),
            nn.Linear(hidden_size_ffn, embed_dim)
        )

    def forward(self, X):
        # Multi-Head Attention
        for i in range(self.head_num):
            Q = X @ self.WQ[i]
            K = X @ self.WK[i]
            V = X @ self.WV[i]
            att = (torch.softmax(Q @ K.transpose(2, 1), dim=1)) / torch.sqrt(self.hidden_size_att)
            vec = att @ V
            if i == 0:
                att_vec = vec
            else:
                att_vec = torch.cat((att_vec, vec), dim=2)
        vector_att = att_vec @ self.WO
        # Add&Norm
        vector_att = self.norm(vector_att + X)
        # Position-wise FFN
        vector_ffn = self.ffn(vector_att)
        # Add&Norm
        vector = self.norm(vector_ffn + vector_att)
        return vector