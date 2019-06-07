#coding=utf-8
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.ticker as ticker
import seaborn as sns

import os


def constructG(filename):
    G = nx.DiGraph().to_undirected()
    with open('networks/'+filename+'', 'r') as f:
        for position, line in enumerate(f):
            t = line.strip().split(' ')
            if t[0] != t[1]:
                G.add_edge(t[0], t[1])
    G.name = filename
    return G

def SIR(G, infected, beta, miu):
    N = 1000
    re = 0
    beta = beta*UC
    while N > 0:
        inf = set(infected)
        R = set()
        while len(inf) != 0:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    k = random.uniform(0,1)
                    if k < beta and j not in inf and j not in R:
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 >miu:
                    newInf.append(i)
                else:
                    R.add(i)
            inf = set(newInf)
        re += len(R)+len(inf)
        N -= 1
    return re/1000.0

def betweenness(G):
    return [x[0] for x in sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)]

def degree(G):
    return [x[0] for x in sorted(G.degree(), key=lambda x: x[1], reverse=True)]

def closeness(G):
    return [x[0] for x in sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1], reverse=True)]

def kShell(G):
    return [x[0] for x in sorted(nx.core_number(G).items(), key=lambda x: x[1], reverse=True)]


def VoteRank(G):
    Vote = {}
    avd = 0
    for node in G.nodes():
        Vote[node] = 1
        avd += G.degree(node)
    k = 1.0/(avd/G.number_of_nodes())
    rank = []
    while True:
        Score = {}
        for node in G.nodes():
            if node not in rank:
                Score[node] = 0
                for neighbor in G.neighbors(node):
                    Score[node] += Vote[neighbor]
        MaxNode, MaxValue = max(Score.items(), key=lambda x: x[1])
        if MaxValue > 0:
            rank.append(MaxNode)
            Vote[MaxNode] = 0
            for neighbor in G.neighbors(MaxNode):
                Vote[neighbor] = max([0, Vote[neighbor]-k])
        else:
            break
    remain = []
    for node in G.nodes():
        if node not in rank:
            remain.append((node, G.degree(node)))
    remain =[ x[0] for x in sorted(remain, key=lambda x:x[1], reverse=True)]
    return rank+remain

def NeighborMatrix(G, k):
    NM = {}
    data = []
    for node in G.nodes():
        neighbors = [node]
        seeds = [node]
        while len(neighbors) < k:
            new_seeds = set([])
            rank = set([])
            for seed in seeds:
                for neighbor in G.neighbors(seed):
                    if neighbor not in neighbors:
                        rank.add((neighbor, G.degree(neighbor)))
                        new_seeds.add(neighbor)
            rank = sorted(rank, key=lambda x:x[1], reverse=True)
            for neighbor in rank:
                neighbors.append(neighbor[0])
                if len(neighbors) == k:
                    NM[node] = neighbors
                    break
            seeds = new_seeds

    for node in NM.keys():
        subM = nx.adjacency_matrix(G, NM[node]).todense()
        for i in range(k):
            subM[i, i] = G.degree(NM[node][i])
            if i == 0:
                for j in range(1, k):
                    if subM[i, j] == 1:
                        subM[i, j] = G.degree(NM[node][j])
                        subM[j, i] = subM[i, j]
        returnVect = []
        for i in range(k):
            for j in range(k):
                returnVect.append(subM[i, j])
        returnVect = np.array(returnVect)
        Re = returnVect.reshape(1, k, k)
        # returnVect = np.zeros((1, k*k))
        # for i in range(k):
        #     for j in range(k):
        #         returnVect[0, k * i + j] = subM[i, j]
        data.append(Re)
    np.save('data_model_1/' + G.name + '_'+str(k)+'_AM.npy', np.array(data))
    return data


class Dataset(Dataset):
    def __init__(self, name, k, beta):

        self.data = np.load('data_model_1/'+name+'_'+str(k)+'_AM.npy')
        self.label = np.load('data_model_1/'+name+'_'+str(beta)+'_label.npy')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        """Convert ndarrays to Tensors."""
        return torch.from_numpy(data).float(),torch.from_numpy(label).float()

class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,28,28)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=16,  # n_filter
                      kernel_size=5,  # filter size
                      stride=1,  # filter step
                      padding=2  # con2d出来的图片大小不变
                      ),  # output shape (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 2x2采样，output shape (16,14,14)

        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32,7,7)
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.out = nn.Linear(int(32 * k/4 * k/4), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flat (batch_size, 32*7*7)
        output = self.out(x)
        return output

def PredictNodeByCNN(nodes, data, k, beta, TrainName, is_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = CNN(k).to(device)
    # optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # loss_fun
    loss_func = nn.MSELoss()

    # training loop
    if is_train:
        for epoch in range(500):
            for i, (x, y) in enumerate(data):
                batch_x = Variable(x).to(device)
                batch_y = Variable(y).to(device)
                # 输入训练数据
                output = cnn(batch_x)
                # 计算误差
                loss = loss_func(output, batch_y)
                print(epoch + 1, i + 1, loss.item())
                # 清空上一次梯度
                optimizer.zero_grad()
                # 误差反向传递
                loss.backward()
                # 优化器参数更新
                optimizer.step()
                state = {'net':cnn.state_dict(),
                         'optimizer':optimizer.state_dict(),
                         'epoch':epoch+1}
                torch.save(state, 'data_model_1/'+TrainName+'_'+str(k)+'_'+str(beta)+'.pth')
    else:
        checkpoint = torch.load('data_model_1/'+TrainName+'_'+str(k)+'_'+str(beta)+'.pth')
        cnn.load_state_dict(checkpoint['net'])
        dataiter = iter(data)
        data, labels = dataiter.next()
        data = data.to(device)
        outputs = cnn(data)
        re = []
        for i in range(len(nodes)):
            re.append((nodes[i], outputs[i].item()))
        rank = [x[0] for x in sorted(re, key=lambda x: x[1], reverse=True)]
        return rank

def MakeReal(label, nodes):
    real = []
    for i in range(len(nodes)):
        real.append((nodes[i], label[i]))
    real = [x[0] for x in sorted(real, key=lambda x: x[1], reverse=True)]
    return real

def nodesRank(rank):
    re = []
    try:
        for i in range(len(rank)):
            re.append(rank.index(str(i)))
    except:
        for i in range(1, len(rank)+1):
            re.append(rank.index(str(i)))
    return re

def corr(G, k, beta_t, beta, train):
    label = np.load('data_model_1/'+G.name+'_'+str(beta)+'_label.npy')
    nodes = list(G.nodes())
    real = MakeReal(label, nodes)
    test_dataset = Dataset(G.name, k, beta)
    test_loader = DataLoader(test_dataset, batch_size=G.number_of_nodes(),  # 分批次训练
                             shuffle=False)
    rankCNN = PredictNodeByCNN(nodes, test_loader, k, beta_t, train, False)
    df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                       'CNN': np.array(nodesRank(rankCNN), dtype=float)
                       })
    return df.corr('kendall')['real']['CNN']

#在不同训练集上，相关性随K的变化(训练感染概率=真实感染概率=1.5)
def draw_k(networks, r):
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        if i<3:
            G = nx.read_edgelist('networks/'+networks[i])
            G.name = networks[i]
        else:
            G = constructG(networks[i])
        K = [x for x in range(8, 48, 4)]
        BA1000_2 = []
        BA1000_5 = []
        BA1000_10 = []
        for k in K:
            print(k)
            BA1000_2.append(corr(G, k, r, r, 'BA1000_2'))
            BA1000_5.append(corr(G, k, r, r, 'BA1000_5'))
            BA1000_10.append(corr(G, k, r, r, 'BA1000_10'))
        plt.subplot(3, 3, i+1)
        plt.plot(K, BA1000_2, 'r', label='BA1000_2', marker='o')
        plt.plot(K, BA1000_5, 'g', label='BA1000_5', marker='*')
        plt.plot(K, BA1000_10, 'b', label='BA1000_10', marker='')
        plt.title(G.name)
        plt.legend(loc='best', fontsize=14)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('k', fontsize=30)
        plt.ylabel('corr', fontsize=30)
        plt.ylim(-0.3, 1)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.savefig('results/all_k.png')
    plt.show()

def heatmap(networks):
    X = [x / 10.0 for x in range(10, 21)]
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        if i < 3:
            G = nx.read_edgelist('networks/' + networks[i])
            G.name = networks[i]
        else:
            G = constructG(networks[i])
        tempR = []
        for j in X:
            for k in X:
                tempR.append(corr(G, 28, j, k, 'BA1000_2'))# 28, 'BA1000_2'
        plt.subplot(3, 3, i+1)
        y = np.array(tempR).reshape((10, 10))
        df = pd.DataFrame(y)
        sns.heatmap(df, annot=False, vmin=0, vmax=1, xticklabels=X, yticklabels=X)
        plt.title(G.name, fontsize=20)
        plt.xlabel(r'$\mu/\mu_{c}$', fontsize=20)
        plt.ylabel(r'$\mu_{t}/\mu_{c}$', fontsize=20)

    plt.show()




if __name__ == '__main__':
    # K = [x for x in range(8, 48, 4)]
    X = [x / 10.0 for x in range(10, 21)]
    networks = ['BA2000_2', 'BA2000_5', 'BA2000_10',
                'Jazz', 'Email', 'Oz',
                'router', 'faa', 'facebook'
                ]

    # G = constructG('facebook')
    # G = nx.read_edgelist('networks/BA1000_2')
    # G.name = 'BA1000_2'
    # print(nx.is_connected(G))
    # print(G.number_of_nodes(), G.number_of_edges(), G.number_of_edges() * 2 / G.number_of_nodes())
    for i in range(len(networks)):
        if i<3:
            G = nx.read_edgelist('networks/'+networks[i])
            G.name = networks[i]
        else:
            G = constructG(networks[i])
        k1 = 0.0
        k2 = 0.0
        for i in G.degree():
            k1 = k1 + i[1]
            k2 = k2 + i[1] ** 2
        global UC
        UC = k1 / (k2 - k1)

        for x in X:
            label = []
            for node in G.nodes():
                label.append([SIR(G, [node], x, 1)])
            np.save('data_model_1/' + G.name + '_' + str(x) + '_label.npy', np.array(label))
        print('标签生成成功')
    #


    # 生成矩阵
    # for k in K:
    #     NeighborMatrix(G, k)
    # print('矩阵生成成功')

    #生成标签
    # for x in X:
    #     label = []
    #     for node in G.nodes():
    #         label.append([SIR(G, [node], x, 1)])
    #     np.save('data_model_1/' + G.name + '_' + str(x) + '_label.npy', np.array(label))
    # print('标签生成成功')

    # Train
    # import time
    # tt = time.time()
    # for x in X:
    #     train_dataset = Dataset(G.name, 28, x)
    #     train_loader = DataLoader(train_dataset, batch_size=G.number_of_nodes(), shuffle=True)
    #     nodes = list(G.nodes())
    #     PredictNodeByCNN(nodes, train_loader, 28, x, G.name, True)
    # print('训练结束')
    # print(time.time()-tt)

    # for k in K:
    #     print(corr(G, k, 1.5, 'BA1000_2'))

    # draw_k(networks, 1.5)

    # 基准方法
    # print(corr(G, 1, 1.5, 'BA1000_2'))


