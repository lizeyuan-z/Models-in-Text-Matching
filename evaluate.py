import numpy as np
import torch

from Model.ABCNN import ABCNN
from Model.ESIM import ESIM

torch.manual_seed(1024)


def evaluate(model_name, model_setup, data_setup, device, batch_size, mode, vocab_size, sentence_length, embed_dim, kernel_width):
    print('开始测试...')
    if model_name == 'ESIM':
        model = ESIM(mode, device, batch_size, vocab_size, sentence_length).to(device)
    elif model_name == 'ABCNN':
        model = ABCNN(device, batch_size, vocab_size, embed_dim, kernel_width).to(device)
    model.load_state_dict(torch.load(model_setup))
    model.eval()
    print('模型加载完毕！')
    dev_index = np.load(data_setup[0], allow_pickle=True).tolist()
    dev_label = np.load(data_setup[1], allow_pickle=True).tolist()
    dev_index = torch.tensor(dev_index, dtype=torch.int64).to(device)
    dev_index_1, dev_index_2 = dev_index.permute(1, 0, 2)[0].to(device), dev_index.permute(1, 0, 2)[1].to(device)
    print('数据加载完毕！')
    predict_label= []
    for i in range(0, len(dev_label), batch_size):
        if i + batch_size <= len(dev_label):
            predict_label = predict_label + torch.max(model(dev_index_1[i:i+batch_size], dev_index_2[i:i+batch_size]), dim=1)[1].tolist()
        else:
            predict_label = predict_label + torch.max(model(dev_index_1[i:i+batch_size], dev_index_2[i:i+batch_size]), dim=1)[1].tolist()
    print(accuracy(predict_label, dev_label))



def accuracy(predict_label,label):
    true_positive = 0
    true_negative = 0
    for i,j in zip(predict_label,label):
        if i == 1 and j == 1:
            true_positive += 1
        if i == 0 and j == 0:
            true_negative += 1
    return (true_positive+true_negative)/len(label)


if __name__ == "__main__":
    model_name = 'ABCNN'
    model_setup = 'trained_model/ABCNN/ABCNN'
    data_setup = ['data/devset/dev_index.npy', 'data/devset/dev_label.npy']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    mode = 'bi'
    vocab_size = 7901
    sentence_length = 15
    embed_dim= 300
    kernel_width = 3
    output = evaluate(model_name, model_setup, data_setup, device, batch_size, mode, vocab_size, sentence_length, embed_dim, kernel_width)