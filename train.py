import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from Model.ABCNN import ABCNN
from Model.BiMPM import BiMPM
from Model.ConvNet import ConvNet
from Model.DRCN import DRCN
from Model.DSSM import DSSM
from Model.ESIM import ESIM
from data.load import data_loader_with_label
from urls import get_index

torch.manual_seed(1024)
torch.cuda.current_device()
torch.cuda._initialized = True


def train(model_name, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path):
    print('模型初始化...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    if model_name == 'DSSM':
        model = DSSM(mode, vocab_setup[1], vocab_setup[0])
        loss_fn = nn.BCELoss()
    elif model_name == 'ESIM':
        model = ESIM(mode, device, batch_size, vocab_setup[1], sentence_length)
    elif model_name == 'ConvNet':
        model = ConvNet(vocab_setup[1], embed_dim, )
    elif model_name == 'ABCNN':
        model = ABCNN(device, batch_size, vocab_setup[1], embed_dim, kernel_width)
    elif model_name == 'BiMPM':
        model = BiMPM(mode, device, vocab_setup[1], sentence_length, perspective_num)
    elif model_name == 'DRCN':
        model = DRCN(device, batch_size, vocab_setup[1], embed_dim)
    elif model_name == 'DAM':
        pass
    elif model_name == 'DIIN':
        pass
    else:
        Warning('模型不在库中或模型名称拼写错误！')

    model.to(device)
    loss_fn.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer.zero_grad()

    print('处理数据集...')
    train_set, train_label = data_loader_with_label(data_setup[0], data_setup[1])
    print('successfully load!')
    # train_set_index = get_index(train_set, vocab_setup[0])
    train_set_index = np.load(data_setup[2], allow_pickle=True).tolist()
    train_label = np.load(data_setup[3], allow_pickle=True).tolist()
    print('train set successfully convert!')
    print('train label successfully cnvert!')

    print('开始训练...')
    if model_name == 'DSSM':
        train_set_index = torch.tensor(train_set_index, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.float).to(device)
        with open('log/dssm_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        j = 0
        for i in range(len(train_set)):
                score = model(train_set[i][0], train_set[i][1], device).to(device)
                loss = loss_fn(score.reshape(1), train_label[i].reshape(1)).to(device)
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad()
                print('>>>batch:{0}    socre:{1}     loss:{2}'.format(i, score, loss))
                with open('log/dssm_log.txt', 'a', encoding='utf-8') as f:
                    f.write('>>>batch:{0}    socre:{1}     loss:{2}\n'.format(i, score, loss))
                j += 1
        print('count:', j)
    elif model_name == 'ESIM':
        train_set_index = torch.tensor(train_set_index, dtype=torch.int64).to(device)
        train_label = torch.tensor(train_label, dtype=torch.int64).to(device)
        train_set_index_1, train_set_index_2 = train_set_index.permute(1, 0, 2)[0].to(device), train_set_index.permute(1, 0, 2)[1].to(device)
        with open('log/esim_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        for i in range(0, len(train_set_index), batch_size):
            if i + batch_size <= len(train_set_index):
                output = model(train_set_index_1[i:i+batch_size], train_set_index_2[i:i+batch_size]).to(device)
                loss = loss_fn(output, train_label[i:i+batch_size]).to(device)
            else:
                output = model(train_set_index_1[i:], train_set_index_2[i:]).to(device)
                loss = loss_fn(output, train_label[i:]).to(device)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            print('>>>batch:{0}     loss:{1}'.format(i / batch_size, float(loss)))
            with open('log/esim_log.txt', 'a', encoding='utf-8') as f:
                f.write('>>>batch:{0}     loss:{1}\n'.format(i / batch_size, float(loss)))
    elif model_name == 'ConvNet':
        train_set_index = torch.tensor(train_set_index, dtype=torch.long)
        train_label = torch.tensor(train_label, dtype=torch.long).to(device)
        train_set_index_1, train_set_index_2 = train_set_index.permute(1, 0, 2)[0].to(device), train_set_index.permute(1, 0, 2)[1].to(device)
        with open('log/convnet_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        for i in range(0, len(train_set_index), batch_size):
            if i + batch_size <= len(train_set_index):
                output = model(train_set_index_1[i:i + batch_size], train_set_index_2[i:i + batch_size]).to(device)
                loss = loss_fn(output, train_label[i:i+batch_size]).to(device)
            else:
                output = model(train_set_index_1[i:], train_set_index_2[i:]).to(device)
                loss = loss_fn(output, train_label[i:]).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            print('>>>batch:{0}     loss:{1}'.format(i / batch_size, float(loss)))
            with open('log/convnet_log.txt', 'a', encoding='utf-8') as f:
                f.write('batch:{0}     loss:{1}\n'.format(i / batch_size, float(loss)))
    elif model_name == 'ABCNN':
        train_set_index = torch.tensor(train_set_index, dtype=torch.long)
        train_label = torch.tensor(train_label, dtype=torch.long).to(device)
        train_set_index_1, train_set_index_2 = train_set_index.permute(1, 0, 2)[0].to(device), train_set_index.permute(1, 0, 2)[1].to(device)
        with open('log/abcnn_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        for i in range(0, len(train_set_index), batch_size):
            if i + batch_size <= len(train_set_index):
                output = model(train_set_index_1[i:i+batch_size], train_set_index_2[i:i+batch_size]).to(device)
                loss = loss_fn(output, train_label[i:i+batch_size]).to(device)
            else:
                output = model(train_set_index_1[i:], train_set_index_2[i:]).to(device)
                loss = loss_fn(output, train_label[i:]).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            print('>>>batch:{0}     loss:{1}'.format(i / batch_size, float(loss)))
            with open('log/abcnn_log.txt', 'a', encoding='utf-8') as f:
                f.write('batch:{0}     loss:{1}\n'.format(i / batch_size, float(loss)))
    elif model_name == 'BiMPM':
        train_set_index = torch.tensor(train_set_index, dtype=torch.long)
        train_label = torch.tensor(train_label, dtype=torch.long).to(device)
        train_set_index_1, train_set_index_2 = train_set_index.permute(1, 0, 2)[0].to(device), train_set_index.permute(1, 0, 2)[1].to(device)
        with open('log/bimpm_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        for i in range(0, len(train_set_index), batch_size):
            if i + batch_size <= len(train_set_index):
                output = model(train_set_index_1[i:i+batch_size], train_set_index_2[i:i+batch_size]).to(device)
                loss = loss_fn(output, train_label[i:i+batch_size]).to(device)
            else:
                output = model(train_set_index_1[i:], train_set_index_2[i:]).to(device)
                loss = loss_fn(output, train_label[i:]).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            print('>>>batch:{0}     loss:{1}'.format(i / batch_size, float(loss)))
            with open('log/bimpm_log.txt', 'a', encoding='utf-8') as f:
                f.write('batch:{0}     loss:{1}\n'.format(i / batch_size, float(loss)))
    elif model_name == 'DRCN':
        train_set_index = torch.tensor(train_set_index, dtype=torch.long)
        train_label = torch.tensor(train_label, dtype=torch.long).to(device)
        train_set_index_1, train_set_index_2 = train_set_index.permute(1, 0, 2)[0].to(device), train_set_index.permute(1, 0, 2)[1].to(device)
        with open('log/drcn_log.txt', 'a', encoding='utf-8') as f:
            f.write('-----------------------------------------------------------------------------------------------\n')
        for i in range(0, len(train_set_index), batch_size):
            if i + batch_size <= len(train_set_index):
                output = model(train_set_index_1[i:i+batch_size], train_set_index_2[i:i+batch_size]).to(device)
                loss = loss_fn(output, train_label[i:i+batch_size]).to(device)
            else:
                output = model(train_set_index_1[i:], train_set_index_2[i:]).to(device)
                loss = loss_fn(output, train_label[i:]).to(device)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            print('>>>batch:{0}     loss:{1}'.format(i / batch_size, float(loss)))
            with open('log/drcn_log.txt', 'a', encoding='utf-8') as f:
                f.write('batch:{0}     loss:{1}\n'.format(i / batch_size, float(loss)))
    elif model_name == 'DAM':
        pass
    elif model_name == 'DIIN':
        pass
    else:
        Warning('模型不在库中或模型名称拼写错误！')
    try:
        torch.save(model, save_path + '/{0}_model.pkl'.format(model_name))
    except:
        torch.save(model, 'trained_model/{0}_model.pkl'.format(model_name))
    try:
        torch.save(model.state_dict(),  save_path + '/{0}.json'.format(model_name) )
    except:
        torch.save(model.state_dict(), save_path)


"""
    data_set: data_path,sep_char
    vocab_setup: vocab_path,vocab_size
"""


if __name__ == "__main__":
    """
    model = 'DSSM'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    mode = 'embed'
    learning_rate = 0.0005
    batch_size = 32
    embed_dim = None
    kernel_width = None
    sentence_length = 15
    perspective_num = None
    save_path = '/home/lizeyuan/NLP/TextMatching/Base/trained_model'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path)
    """

    """
    model = 'ESIM'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    mode = 'bi'
    learning_rate = 0.0004
    batch_size = 32
    embed_dim = None
    kernel_width = None
    sentence_length = 15
    perspective_num = None
    save_path = 'trained_model/ESIM' # '/home/lizeyuan/NLP/TextMatching/Base/trained_model/ESIM'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path)
    """

    """
    model = 'ConvNet'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    mode = None
    learning_rate = 0.0004
    batch_size = 32
    embed_dim = None
    kernel_width = None
    sentence_length = 15
    save_path = '/home/lizeyuan/NLP/TextMatching/Base/trained_model/ConvNet'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, save_path)
    """

    """
    model = 'ABCNN'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    mode = None
    learning_rate = 0.001
    batch_size = 32
    embed_dim = 300
    kernel_width = 3
    sentence_length = 15
    perspective_num = None
    save_path = '/home/lizeyuan/NLP/TextMatching/Base/trained_model/ABCNN'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path)
    """


    model = 'BiMPM'
    mode = 'maxpooling match'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    learning_rate = 0.001
    batch_size = 32
    embed_dim = 100
    kernel_width = None
    sentence_length = 15
    perspective_num = 12
    save_path = '/home/lizeyuan/NLP/TextMatching/Base/trained_model/BiMPM'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path)


    """
    model = 'DRCN'
    data_setup = ['data/trainset/train.csv', '\t', 'data/trainset/train_index.npy', 'data/trainset/train_label.npy']
    vocab_setup = ['Lookup/vocab/vocab1.txt', 7901]
    mode = None
    learning_rate = 0.0002
    batch_size = 32
    embed_dim = 300
    kernel_width = None
    sentence_length = 15
    perspective_num = None
    save_path = '/home/lizeyuan/NLP/TextMatching/Base/trained_model/BiMPM'
    train(model, data_setup, vocab_setup, mode, learning_rate, batch_size, embed_dim, kernel_width, sentence_length, perspective_num, save_path)
    """
