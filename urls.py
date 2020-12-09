import torch

from Lookup.lookup import LookUp


def get_index(train_set, vocab_path):
    lookup = LookUp(vocab_path)
    train_set_index = train_set
    for i in range(len(train_set)):
        for j in range(len(train_set[0])):
            tmp = lookup.get_index(train_set[i][j])
            if len(tmp) >= 15:
                tmp = tmp[:15]
            else:
                tmp = tmp + [0] * (15-len(tmp))
            train_set_index[i][j] = tmp
    return train_set_index
