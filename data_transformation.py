import numpy as np

from Lookup.lookup import LookUp


path = "data/devset/dev.csv"
vocab_path = 'Lookup/vocab/vocab1.txt'
train_set = []
train_label = []
with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split('\t')
            train_set.append(tmp[:2])
            train_label.append(int(tmp[2]))
lookup = LookUp(vocab_path)
train_set_index = []
for i in range(len(train_set)):
    line = []
    for j in range(len(train_set[0])):
        tmp = lookup.get_index(train_set[i][j])
        if len(tmp) < 15:
            tmp = tmp + [0] * (15 - len(tmp))
        if len(tmp) >= 15:
            tmp = tmp[:15]
        line.append(tmp)
    train_set_index.append(line)
data = np.array(train_set_index)
np.save('data/devset/dev_index', data)
data = np.array(train_label)
np.save('data/devset/dev_label', data)
