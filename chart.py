import matplotlib
import matplotlib.pyplot as plt

batch = []
loss = []
with open('tmp_file/xab.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        batch.append(int(tmp[0]))
        loss.append(tmp[1])

plt.plot(batch, loss)
