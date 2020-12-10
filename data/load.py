def data_loader_with_label(data_path, sep_char):
    train_set = []
    train_label = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split(sep_char)
            if len(tmp[0]) < 15:
                tmp[0] = tmp[0] + '[PAD]' * (15 - len(tmp[0]))
            else:
                tmp[0] = tmp[0][0:15]
            if len(tmp[1]) < 15:
                tmp[1] = tmp[1] + '[PAD]' * (15 - len(tmp[1]))
            else:
                tmp[1] = tmp[1][0:15]
            train_set.append(tmp[:2])
            train_label.append(int(tmp[2]))
    return train_set, train_label
