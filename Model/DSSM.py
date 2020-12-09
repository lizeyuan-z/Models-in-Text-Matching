import torch.nn as nn
import torch
from Lookup.lookup import LookUp


class DSSM(nn.Module):
    def __init__(self, mode, vocab_size, corpus, embed_dim=30621):
        super(DSSM, self).__init__()
        self.mode = mode

        if self.mode == 'word hashing':
            self.letter_trigram = WordHashing(corpus)
            self.input_vector_dim = len(self.letter_trigram.lettertrigram)
        elif self.mode == 'embed':
            self.lookup = LookUp(corpus)
            self.input_vector_dim = embed_dim
            self.embed = nn.Embedding(vocab_size, self.input_vector_dim)
        else:
            Warning('mode必须为word hashing 或 embed ！')

        self.dense1 = nn.Linear(self.input_vector_dim, 300)
        self.dense2 = nn.Linear(300, 300)
        self.dense3 = nn.Linear(300, 128)

    def forward(self, query, doc, device):
        if self.mode == 'word hashing':
            query_vec = self.dense3(self.dense2(self.dense1(self.letter_trigram.word_hashing(BagOfWords(query).word_count).tanh()).tanh()).tanh()).tanh()
            doc_vec = self.dense3(self.dense2(self.dense1(self.letter_trigram.word_hashing(BagOfWords(doc).word_count).tanh()).tanh()).tanh()).tanh()
            score = torch.cosine_similarity(query_vec, doc_vec, dim=0)
            return score
        elif self.mode == 'embed':
            query_vec = self.dense3(self.dense2(self.dense1(self.embed(torch.LongTensor(self.lookup.get_index(query)).to(device)).tanh()).tanh()).tanh()).tanh()
            doc_vec = self.dense3(self.dense2(self.dense1(self.embed(torch.LongTensor(self.lookup.get_index(doc)).to(device)).tanh()).tanh()).tanh()).tanh()
            query_vec = query_vec.reshape(len(query_vec) * len(query_vec[0]))
            doc_vec = doc_vec.reshape(len(doc_vec) * len(doc_vec[0]))
            score = torch.cosine_similarity(query_vec[0:min(len(query_vec), len(doc_vec))], doc_vec[0:min(len(query_vec), len(doc_vec))], dim=0).to(device)
            return score.sigmoid()
        else:
            Warning('mode必须为word hashing 或 embed ！')


class BagOfWords(nn.Module):
    def __init__(self, X, min_count=0, max_len=100):
        super(BagOfWords, self).__init__()
        self.X = X
        self.min_count = min_count
        self.max_len = max_len
        self.__word_count()
        self.__doc2num()
        self.__index()

    def __word_count(self):
        word_count = {}
        for word_sequence in self.X:
            for word in word_sequence:
                if word in word_count.keys():
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        self.word_count = {i: j for i, j in word_count.items() if j >= self.min_count}

    def __index(self):
        self.index2word = {i + 1: j for i, j in enumerate(self.word_count)}
        self.word2index = {j: i for i, j in self.index2word.items()}

    def __doc2num(self):
        self.doc2num = []
        for text in self.X:
            s = [self.word2index.get(i, 0) for i in text[:self.max_len]]
            self.doc2num.append(s + [0]*(self.max_len - len(s)))


class WordHashing(nn.Module):
    def __init__(self, corpus):
        super(WordHashing, self).__init__()
        self.corpus = corpus
        self.letter_trigram()

    def letter_trigram(self):
        self.lettertrigram = []
        for word in self.corpus.keys():
            word = '#' + word + '#'
            for i in range(len(word)-2):
                if word[i:i+3] not in self.lettertrigram:
                    self.lettertrigram.append(word[i:i+3])

    def word_hashing(self,bag_of_words):
        self.vector = [0]*len(self.lettertrigram)
        for word in bag_of_words.keys():
            for trigram in self.lettertrigram:
                if trigram in word:
                    self.vector[self.lettertrigram.index(trigram) + 1] += bag_of_words[word]
        return torch.LongTensor(self.vector)
