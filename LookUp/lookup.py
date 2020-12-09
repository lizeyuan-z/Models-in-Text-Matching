class LookUp(object):
    def __init__(self, vocab_path):
        self.__get_vocab(vocab_path)

    def get_index(self, sentence):
        output = []
        for word in sentence:
            if word in self.vocab:
                output.append(self.vocab.index(word))
            else:
                output.append(1)
        return output

    def __get_vocab(self, vocab_path):
        self.vocab = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                tmp = line.strip()
                self.vocab.append(tmp)
