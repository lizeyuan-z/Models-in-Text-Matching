import torch
import torch.nn as nn


class BiMPM(nn.Module):
    def __init__(self, mode, device, vocab_size, sentence_length, perspective_num, embed_dim=300, hidden_size=100):
        super(BiMPM, self).__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm1_1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, dropout=0.1, bidirectional=True)
        self.lstm1_2 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True, dropout=0.1, bidirectional=True)
        self.lstm2_1_forward = nn.LSTM(input_size=perspective_num, hidden_size=perspective_num, batch_first=True, dropout=0.1, bidirectional=False)
        self.lstm2_2_forward = nn.LSTM(input_size=perspective_num, hidden_size=perspective_num, batch_first=True, dropout=0.1, bidirectional=False)
        self.lstm2_1_backward = nn.LSTM(input_size=perspective_num, hidden_size=perspective_num, batch_first=True, dropout=0.1, bidirectional=False)
        self.lstm2_2_backward = nn.LSTM(input_size=perspective_num, hidden_size=perspective_num, batch_first=True, dropout=0.1, bidirectional=False)
        self.W1 = nn.Parameter(torch.randn((perspective_num, hidden_size * 2), requires_grad=True))
        self.W2 = nn.Parameter(torch.randn((perspective_num, hidden_size * 2), requires_grad=True))
        self.match = Matcher(device)
        self.dnn = nn.Sequential(
            nn.Linear(perspective_num * 4, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, sentence1, sentence2):
        sentence1_encoded, _ = self.lstm1_1(self.embed(sentence1))
        sentence2_encoded, _ = self.lstm1_2(self.embed(sentence2))
        if self.mode == 'full matching':
            for i in range(len(sentence1)):
                sentence1_forward, sentence1_backward = self.match.full_matching(sentence1_encoded[i][-1],
                                                                                 sentence1_encoded[i][-1],
                                                                                 sentence2_encoded[i],
                                                                                 self.embed_dim, self.W1, self.W2)
                sentence2_forward, sentence2_backward = self.match.full_matching(sentence2_encoded[i][-1],
                                                                                 sentence2_encoded[i][-1],
                                                                                 sentence1_encoded[i],
                                                                                 self.embed_dim, self.W1, self.W2)
                if i == 0:
                    sentence1_matched_forward = sentence1_forward.unsqueeze(0)
                    sentence1_matched_backward = sentence1_backward.unsqueeze(0)
                    sentence2_matched_forward = sentence2_forward.unsqueeze(0)
                    sentence2_matched_backward = sentence2_backward.unsqueeze(0)
                else:
                    sentence1_matched_forward = torch.cat((sentence1_matched_forward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence1_matched_backward = torch.cat((sentence1_matched_backward, sentence1_backward.unsqueeze(0)), dim=0)
                    sentence2_matched_forward = torch.cat((sentence2_matched_forward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_backward = torch.cat((sentence2_matched_backward, sentence1_backward.unsqueeze(0)), dim=0)
        elif self.mode == 'maxpooling match':
            for i in range(len(sentence1)):
                sentence1_forward, sentence1_backward = self.match.maxpooling_matching(sentence1_encoded[i], sentence2_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                sentence2_forward, sentence2_backward = self.match.maxpooling_matching(sentence2_encoded[i], sentence1_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                if i == 0:
                    sentence1_matched_forward = sentence1_forward.unsqueeze(0)
                    sentence1_matched_backward = sentence1_backward.unsqueeze(0)
                    sentence2_matched_forward = sentence2_forward.unsqueeze(0)
                    sentence2_matched_backward = sentence2_backward.unsqueeze(0)
                else:
                    sentence1_matched_forward = torch.cat((sentence1_matched_forward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence1_matched_backward = torch.cat((sentence1_matched_backward, sentence1_backward.unsqueeze(0)), dim=0)
                    sentence2_matched_forward = torch.cat((sentence2_matched_forward, sentence2_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_backward = torch.cat((sentence2_matched_backward, sentence2_backward.unsqueeze(0)), dim=0)
        elif self.mode == 'attentive matching':
            for i in range(len(sentence1)):
                sentence1_forward, sentence1_backward = self.match.maxpooling_matching(sentence1_encoded[i], sentence2_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                sentence2_forward, sentence2_backward = self.match.maxpooling_matching(sentence2_encoded[i], sentence1_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                if i == 0:
                    sentence1_matched_forward = sentence1_forward.unsqueeze(0)
                    sentence1_matched_backward = sentence1_forward.unsqueeze(0)
                    sentence2_matched_forward = sentence2_forward.unsqueeze(0)
                    sentence2_matched_backward = sentence2_forward.unsqueeze(0)
                else:
                    sentence1_matched_forward = torch.cat((sentence1_matched_forward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence1_matched_backward = torch.cat((sentence1_matched_backward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_forward = torch.cat((sentence2_matched_forward, sentence2_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_backward = torch.cat((sentence2_matched_backward, sentence2_forward.unsqueeze(0)), dim=0)
        elif self.mode == 'max attentive pooling matching':
            for i in range(len(sentence1)):
                sentence1_forward, sentence1_backward = self.match.maxpooling_matching(sentence1_encoded[i], sentence2_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                sentence2_forward, sentence2_backward = self.match.maxpooling_matching(sentence2_encoded[i], sentence1_encoded[i],
                                                                                       self.embed_dim, self.W1, self.W2)
                if i == 0:
                    sentence1_matched_forward = sentence1_forward.unsqueeze(0)
                    sentence1_matched_backward = sentence1_backward.unsqueeze(0)
                    sentence2_matched_forward = sentence2_forward.unsqueeze(0)
                    sentence2_matched_backward = sentence2_backward.unsqueeze(0)
                else:
                    sentence1_matched_forward = torch.cat((sentence1_matched_forward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence1_matched_backward = torch.cat((sentence1_matched_backward, sentence1_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_forward = torch.cat((sentence2_matched_forward, sentence2_forward.unsqueeze(0)), dim=0)
                    sentence2_matched_backward = torch.cat((sentence2_matched_backward, sentence2_forward.unsqueeze(0)), dim=0)
        else:
            Warning('match 必须为 full matching 或 maxpooling match 或 attentive matching 或 max attentive pooling matching')
        vec1 = self.lstm2_1_forward(sentence1_matched_forward)[0].permute(1, 0, 2)[-1]
        vec2 = self.lstm2_1_backward(sentence1_matched_backward)[0].permute(1, 0, 2)[-1]
        vec3 = self.lstm2_2_forward(sentence2_matched_forward)[0].permute(1, 0, 2)[-1]
        vec4 = self.lstm2_2_backward(sentence2_matched_backward)[0].permute(1, 0, 2)[-1]
        vector = torch.cat((vec1, vec2, vec3, vec4), dim=1).softmax(dim=1)
        output = self.dnn(vector)
        return output


class Matcher(nn.Module):
    def __init__(self, device):
        super(Matcher, self).__init__()
        self.device = device

    def full_matching(self, forward, backward, sentence2, embed_dim, W1, W2):
        m_forward_full = torch.zeros([len(sentence2), len(W1)]).to(self.device)
        m_backward_full = torch.zeros([len(sentence2), len(W2)]).to(self.device)
        for i in range(len(sentence2)):
            for k in range(len(W1)):
                m_forward_full[i][k] = torch.cosine_similarity(forward * W1[k], sentence2[i] * W1[k], dim=0)
                m_backward_full[i][k] = torch.cosine_similarity(backward * W2[k], sentence2[i] * W2[k], dim=0)
        return m_forward_full, m_backward_full

    def maxpooling_matching(self, sentence1, sentence2, embed_dim, W1, W2):
        matrix_forward = torch.Tensor([[[0] * len(W1)] * len(sentence2)] * len(sentence1)).to(self.device)
        matrix_backward = torch.Tensor([[[0] * len(W2)] * len(sentence2)] * len(sentence1)).to(self.device)
        for i in range(len(sentence1)):
            for j in range(len(sentence2)):
                for k in range(len(W1)):
                    matrix_forward[i][j][k] = torch.cosine_similarity(sentence1[i] * W1[k], sentence2[j] * W1[k], dim=0)
                    matrix_backward[i][j][k] = torch.cosine_similarity(sentence1[i] * W2[k], sentence2[j] * W2[k], dim=0)
        matrix_forward = torch.max(matrix_forward,dim=1)[0]
        matrix_backward = torch.max(matrix_backward, dim=1)[0]
        return matrix_forward, matrix_backward

    def attentive_matching(self, sentence1, sentence2, embed_dim, W1, W2):
        attention_forward = torch.Tensor([[[0] * len(W1)] * len(sentence2)] * len(sentence1)).to(self.device)
        attention_backward = torch.Tensor([[[0] * len(W2)] * len(sentence2)] * len(sentence1)).to(self.device)
        for i in range(len(sentence1)):
            for j in range(len(sentence2)):
                for k in range(len(W1)):
                    attention_forward[i][j][k] = torch.cosine_similarity(sentence1[i] * W1[k], sentence2[j] * W1[k], dim=0)
                    attention_backward[i][j][k] = torch.cosine_similarity(sentence1[i] * W2[k], sentence2[j] * W2[k], dim=0)
        h_mean_forward = (attention_forward @ sentence2) / torch.sum(attention_forward, dim=2).unsqueeze(2)
        h_mean_backward = (attention_backward @ sentence2) / torch.sum(attention_backward, dim=2).unsqueeze(2)


        matrix_forward = torch.Tensor([[0 * len(W1)]] * len(sentence1)).to(self.device)
        matrix_backward = torch.Tensor([[0] * len(W2)] * len(sentence1)).to(self.device)
        for i in range(len(sentence1)):
            for k in range(W1):
                matrix_forward[i][k] = torch.cosine_similarity(sentence1[i] * W1[k], h_mean_forward[i] * W1[k], dim=0)
                matrix_backward[i][k] = torch.cosine_similarity(sentence1[i] * W2[k], h_mean_backward[i] * W2[k], dim=0)
        return matrix_forward, matrix_backward

    def max_attentive_pooling(self, sentence1, sentence2, embed_dim, W1, W2):
        attention_forward = torch.Tensor([[0] * len(sentence2)] * len(sentence1)).to(self.device)
        attention_backward = torch.Tensor([[0] * len(sentence2)] * len(sentence1)).to(self.device)
        for i in range(len(sentence1)):
            for j in range(len(sentence2)):
                attention_forward[i][j] = torch.cosine_similarity(sentence1[i][:embed_dim] * W1[i], sentence2[j][:embed_dim] * W1[j], dim=0)
                attention_backward[i][j] = torch.cosine_similarity(sentence1[i][embed_dim:] * W2[i], sentence2[j][embed_dim:] * W2[j], dim=0)
        h_mean_forward = (attention_forward @ sentence2) / torch.sum(attention_forward, dim=1).unsqueeze(1)
        h_mean_backward = (attention_backward @ sentence2) / torch.sum(attention_backward, dim=1).unsqueeze(1)

        matrix_forward = torch.Tensor([0] * len(sentence1)).to(self.device)
        matrix_backward = torch.Tensor([0] * len(sentence1)).to(self.device)
        for i in range(len(sentence1)):
            matrix_forward[i] = torch.cosine_similarity(sentence1[i] * W1[i], h_mean_forward[i] * W1[i], dim=0)
            matrix_backward[i] = torch.cosine_similarity(sentence1[i] * W2[i], h_mean_backward[i] * W2[i], dim=0)
        return matrix_forward, matrix_backward
