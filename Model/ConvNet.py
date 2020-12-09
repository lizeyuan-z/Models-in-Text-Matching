import math

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, corpus, stop_word_path, device, kernel_num=100, kernel_width=5):
        super(ConvNet, self).__init__()
        self.device = device
        self.idf = IDF(corpus)
        self.overlap = OverLap(stop_word_path)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.covn2d = nn.Conv2d(in_channels=1, out_channels=kernel_num, kernel_size=(kernel_width, embed_dim))
        self.pooling = Pooling()
        self.M = nn.Parameter(torch.randn((embed_dim, embed_dim), requires_grad=True))
        self.dnn = nn.Sequential(
            nn.Linear(kernel_num * 2 + 1 + 2, 64),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax()
        )

    def forward(self, query, doc, feat):
        overlap_all = []
        overlap_unstopworded = []
        wighted_overlap_all = []
        wighted_overlap_unstopworded = []
        for q, d in zip(query, doc):
            tmp1, tmp2 = self.overlap.overlap(q, d)
            overlap_all.append(len(tmp1))
            overlap_unstopworded.append(len(tmp2))
            wighted_overlap_all.append(sum(self.idf.compute(tmp1)))
            wighted_overlap_unstopworded.append(sum(self.idf.compute(tmp2)))
        overlap_all = torch.Tensor(overlap_all).to(self.device)
        wighted_overlap_all = torch.Tensor(wighted_overlap_all).to(self.device)
        feat = torch.cat((overlap_all.reshape(len(overlap_all), 1), wighted_overlap_all.reshape(len(overlap_all), 1)), dim=1)
        query_encode = self.pooling.pooling(self.covn2d(self.embed(query)))
        doc_encode = self.pooling.pooling(self.covn2d(self.embed(doc)))
        similarity = (torch.matmul(torch.matmul(query_encode,self.M),
                                  doc_encode.t())*torch.eye(len(query_encode))
                      ).max(dim=1)[0]
        output = self.dnn(torch.cat((query_encode, doc_encode, similarity, feat), dim=1))
        return output


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def pooling(self, feature_maps):
        return torch.max(feature_maps, dim=2)[0]


class IDF(nn.Module):
    def __init__(self, corpus):
        """
        :param corpus: list([[]])
        """
        super(IDF, self).__init__()
        self.corpus = corpus
        self.__initialize()

    def compute(self, x):
        """
        :param x: overlap words
        :return:
        """
        idf = []
        for key in x:
            idf.append(len(self.corpus)/math.log(len(self.bag_of_words[key]) + 1))
        return idf

    def __initialize(self):
        self.bag_of_words = {}
        for sentence in self.corpus:
            for word in sentence:
                if word not in self.bag_of_words.keys():
                    self.bag_of_words[word] = [self.corpus.index(sentence)]
                else:
                    if self.corpus.index(sentence) not in self.bag_of_words[word]:
                        self.bag_of_words[word].append(self.corpus.index(sentence))


class OverLap(nn.Module):
    def __init__(self, stopword_filepath):
        super(OverLap, self).__init__()
        self.stopword = []
        with open(stopword_filepath) as f:
            for line in f:
                self.stopword.append(line.strip())

    def overlap(self, query, doc):
        self.overlap_all = []
        self.overlap_unstopworded = []
        for i in query:
            if i in doc:
                self.overlap_all.append(i)
                if i not in self.stopword:
                    self.overlap_unstopworded.append(i)
        return self.overlap_all, self.overlap_unstopworded
