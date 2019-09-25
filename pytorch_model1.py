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
    N = 100
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
    return re/100.0

def betweenness(G):
    if G.number_of_nodes()>1000:
        return [x[0] for x in sorted(nx.betweenness_centrality(G, k=100).items(), key=lambda x: x[1], reverse=True)]
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

def H(array):
    narray = sorted(array, reverse=True)
    for i in range(len(narray)):
        if narray[i] < i + 1:
            return i
    return len(narray)
def H_index(G, k):
    N = 1
    HK = {}
    tempDict = {}
    for node in G.nodes():
        tempDict[node] = G.degree(node)
    HK[0] = tempDict
    while N <= k:
        tempDict = {}
        for node in G.nodes():
            tempArray = []
            for neighbor in G.neighbors(node):
                tempArray.append(HK[N-1][neighbor])
            tempDict[node] = H(tempArray)
        HK[N] = tempDict
        N = N + 1
    rank = [x[0] for x in sorted(HK[k].items(), key=lambda x: x[1], reverse=True)]

    return rank

def PPD(G, beta):
    k1 = 0.0
    k2 = 0.0
    for i in G.degree():
        k1 = k1 + i[1]
        k2 = k2 + i[1] ** 2
    UC = k1 / (k2 - k1)
    rank = {}
    for node in G.nodes():
        rank[node] = 0
        N1 = list(G.neighbors(node))
        N2 = []
        N3 = []
        for n1 in N1:
            for neighbor in G.neighbors(n1):
                if neighbor not in N1:
                    N2.append(neighbor)
        for n2 in N2:
            for neighbor in G.neighbors(n2):
                if neighbor not in N1 and neighbor not in N2:
                    N3.append(neighbor)
        SK = {}

        tempDict = {}
        tempDict[node] = 1
        SK[0] = tempDict  # score(u,0)=1

        tempDict = {}
        for n1 in N1:
            tempDict[n1] = beta*UC
        SK[1] = tempDict

        tempDict = {}
        for n2 in N2:
            tempDict[n2] = 1
            for n1 in N1:
                if G.has_edge(n2, n1):
                    tempDict[n2] = tempDict[n2]*(1 - SK[1][n1]*beta*UC)
            tempDict[n2] = 1 - tempDict[n2]
        SK[2] = tempDict

        tempDict = {}
        for n3 in N3:
            tempDict[n3] = 1
            for n2 in N2:
                if G.has_edge(n3, n2):
                    tempDict[n3] = tempDict[n3] * (1 - SK[2][n2] * beta*UC)
            tempDict[n3] = 1 - tempDict[n3]
        SK[3] = tempDict

        for i in range(4):
            for key in SK[i].keys():
                rank[node] += SK[i][key]
    return [x[0] for x in sorted(rank.items(), key=lambda x: x[1], reverse=True)]

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
            if len(seeds) == 0:
                NM[node] = neighbors
                break

    for node in NM.keys():
        subM = nx.adjacency_matrix(G, NM[node]).todense()
        for i in range(len(subM)):
            subM[i, i] = G.degree(NM[node][i])
            if i == 0:
                for j in range(1, len(subM)):
                    if subM[i, j] == 1:
                        subM[i, j] = G.degree(NM[node][j])
                        subM[j, i] = subM[i, j]
        returnVect = []
        for i in range(k):
            for j in range(k):
                if i >= len(subM) or j >= len(subM):
                    returnVect.append(0)
                else:
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
                torch.save(state, 'ttt/'+TrainName+'_'+str(k)+'_'+str(beta)+'.pth')
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
    SR = sorted(rank)
    re = []

    for i in SR:
        re.append(rank.index(i))

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
def draw_k(r):
    networks = ['BA2000_2','BA2000_5','BA2000_10',
                'BA4000_2', 'BA4000_5', 'BA4000_10',
                'BA8000_2', 'BA8000_5', 'BA8000_10'
    ]
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        print(i)
        G = nx.read_edgelist('networks/'+networks[i])
        G.name = networks[i]
        K = [x for x in range(8, 48, 4)]
        BA1000_2 = []
        BA1000_5 = []
        BA1000_10 = []
        BA3000_2 = []
        BA3000_5 = []
        BA3000_10 = []
        for k in K:
            print(k)
            BA1000_2.append(corr(G, k, r, r, 'BA1000_2'))
            BA1000_5.append(corr(G, k, r, r, 'BA1000_5'))
            BA1000_10.append(corr(G, k, r, r, 'BA1000_10'))
            BA3000_2.append(corr(G, k, r, r, 'BA3000_2'))
            BA3000_5.append(corr(G, k, r, r, 'BA3000_5'))
            BA3000_10.append(corr(G, k, r, r, 'BA3000_10'))
        plt.subplot(3, 3, i+1)
        plt.plot(K, BA1000_2, 'r', label='Train_1000_4', marker='o')
        plt.plot(K, BA1000_5, 'b', label='Train_1000_10', marker='*')
        plt.plot(K, BA1000_10, 'g', label='Train_1000_20', marker='^')
        plt.plot(K, BA3000_2, 'r', label='Train_3000_4', marker='v')
        plt.plot(K, BA3000_5, 'b', label='Train_3000_10', marker='<')
        plt.plot(K, BA3000_10, 'g', label='Train_3000_20', marker='>')
        if i == 2:
            plt.legend(loc='best', fontsize=15)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('L', fontsize=25)
        plt.ylabel('corr', fontsize=25)
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.savefig('results/all_k.png')
    plt.show()

def heatmap():
    networks = ['BA2000_2', 'BA2000_5', 'BA2000_10',
                'BA4000_2', 'BA4000_5', 'BA4000_10',
                'BA8000_2', 'BA8000_5', 'BA8000_10'
                ]

    titles = ['Test_2000_4', 'Test_2000_10', 'Test_2000_20',
                'Test_4000_4', 'Test_4000_10', 'Test_4000_20',
                'Test_8000_4', 'Test_8000_10', 'Test_8000_20'
                ]
    X = [x / 10.0 for x in range(10, 21)]
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        G = nx.read_edgelist('networks/' + networks[i])
        G.name = networks[i]
        tempR = []
        for j in X:
            print(j)
            for k in X:
                print(k)
                tempR.append(corr(G, 28, j, k, 'BA1000_2'))# 28, 'BA1000_2'
        plt.subplot(3, 3, i+1)
        y = np.array(tempR).reshape((11, 11))
        df = pd.DataFrame(y)
        sns.heatmap(df, annot=False, vmin=0, vmax=1, xticklabels=X, yticklabels=X)
        plt.title(titles[i], fontsize=20)
        plt.xlabel(r'$\mu/\mu_{c}$', fontsize=25)
        plt.ylabel(r'$\mu_{t}/\mu_{c}$', fontsize=25)
        plt.tick_params(labelsize=20)
        plt.xticks()

    plt.show()

def compareAll():
    networks = ['Email', 'NS', 'USAir',
                'Jazz', 'Oz',
                'Router', 'Faa',
                'figeys','Sex'
         ]
    X = [x / 10.0 for x in range(10, 21)]
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        print(networks[i])
        G = constructG(networks[i])
        nodes = list(G.nodes())
        D = np.array(nodesRank(degree(G)), dtype=float)
        B = np.array(nodesRank(betweenness(G)), dtype=float)
        VR = np.array(nodesRank(VoteRank(G)), dtype=float)
        Kshell = np.array(nodesRank(kShell(G)), dtype=float)
        Hindex = np.array(nodesRank(H_index(G, 1)), dtype=float)
        # try:
        #     DR = np.load('data_model_1/' + G.name + '_DR.npy')
        # except:
        #     DR = np.array(nodesRank(PPD(G, 1.5)), dtype=float)
        #     np.save('data_model_1/' + G.name + '_DR.npy', DR)


        print('xxxx')
        test_dataset = Dataset(G.name, 28, 1.5)
        test_loader = DataLoader(test_dataset, batch_size=G.number_of_nodes(),  # 分批次训练
                                 shuffle=False)
        rankCNN = PredictNodeByCNN(nodes, test_loader, 28, 1.5, 'BA1000_2', False)
        rankCNN = np.array(nodesRank(rankCNN), dtype=float)

        tempD = []
        tempB = []
        tempVR = []
        tempKshell = []
        tempHindex = []
        # tempDR = []
        tempCNN = []

        for j in X:
            print(j)
            label = np.load('data_model_1/' + G.name + '_'+str(j)+'_label.npy')
            real = MakeReal(label, nodes)

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'degree': D
                               })
            tempD.append(df.corr('kendall')['real']['degree'])

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'betweenness': B
                               })
            tempB.append(df.corr('kendall')['real']['betweenness'])

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'VR': VR
                               })
            tempVR.append(df.corr('kendall')['real']['VR'])

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'Kshell': Kshell
                               })
            tempKshell.append(df.corr('kendall')['real']['Kshell'])

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'Hindex': Hindex
                               })
            tempHindex.append(df.corr('kendall')['real']['Hindex'])

            # df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
            #                    'DR': DR
            #                    })
            # tempDR.append(df.corr('kendall')['real']['DR'])

            df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                               'CNN': rankCNN
                               })
            tempCNN.append(df.corr('kendall')['real']['CNN'])
        plt.subplot(3, 3, i + 1)
        plt.plot(X, tempCNN, 'r', label='RCNN', marker='o')
        plt.plot(X, tempB, 'g', label='Betweenness', marker='*')
        #plt.plot(X, tempVR, 'b', label='VoteRank', marker='v')
        plt.plot(X, tempKshell, 'b', label='k-shell', marker='<')
        plt.plot(X, tempD, 'k', label='Degree', marker='>')
        plt.plot(X, tempHindex, 'k', label='H-index', marker='^')
        # plt.plot(X, tempDR, 'g', label='DynamicRank', marker='.')
        if i==0:
            plt.legend(loc='best', fontsize=14)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r'$\mu/\mu_{c}$', fontsize=25)
        plt.ylabel('corr', fontsize=25)
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.show()

def compareAll_topK(k):
    networks = ['Email', 'NS', 'USAir',
                'Jazz', 'Oz',
                'Router', 'Faa',
                'figeys', 'Sex'
                ]
    X = [x / 10.0 for x in range(10, 21)]
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i in range(len(networks)):
        print(networks[i])
        G = constructG(networks[i])

        k1 = 0.0
        k2 = 0.0
        for j in G.degree():
            k1 = k1 + j[1]
            k2 = k2 + j[1] ** 2
        global UC
        UC = k1 / (k2 - k1)

        nodes = list(G.nodes())
        N = G.number_of_nodes()
        D = degree(G)[:int(N*k)]
        B = betweenness(G)[:int(N*k)]
        VR = VoteRank(G)[:int(N*k)]
        Kshell = kShell(G)[:int(N*k)]

        test_dataset = Dataset(G.name, 28, 1.5)
        test_loader = DataLoader(test_dataset, batch_size=G.number_of_nodes(),  # 分批次训练
                                 shuffle=False)
        rankCNN = PredictNodeByCNN(nodes, test_loader, 28, 1.5, 'BA1000_2', False)[:int(N*k)]

        tempD = []
        tempB = []
        tempVR = []
        tempKshell = []
        tempCNN = []

        for j in X:
            print(j)
            tempD.append(SIR(G, D, j, 1)/N)
            tempB.append(SIR(G, B, j, 1)/N)
            tempVR.append(SIR(G, VR, j, 1)/N)
            tempKshell.append(SIR(G, Kshell, j, 1)/N)
            tempCNN.append(SIR(G, rankCNN, j, 1)/N)
        plt.subplot(3, 3, i + 1)
        plt.plot(X, tempCNN, 'r', label='CNN', marker='o')
        plt.plot(X, tempB, 'g', label='Betweenness', marker='*')
        plt.plot(X, tempVR, 'b', label='VoteRank', marker='v')
        plt.plot(X, tempKshell, 'b', label='Kshell', marker='<')
        plt.plot(X, tempD, 'k', label='Degree', marker='>')
        if i==0:
            plt.legend(loc='best', fontsize=14)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r'$\mu/\mu_{c}$', fontsize=25)
        plt.ylabel('Rs', fontsize=25)
        plt.ylim(0, 1)
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.show()

def networkFeat():
    networks = ['NS', 'USAir',
                'Jazz', 'Email', 'Oz',
                'Router', 'Faa',
                'figeys', 'Sex'
                ]
    for i in range(len(networks)):
        print(networks[i])
        G = constructG(networks[i])
        print(G.number_of_nodes(),G.number_of_edges())
        k=float(G.number_of_edges()*2)/G.number_of_nodes()
        print(k)#平均度
        print(list(sorted(G.degree(),key=lambda x:x[1],reverse=True)[0])[1])
        print(nx.average_clustering(G))#网络聚集系数
        k2=0.0
        temp = G.degree()
        for i in temp:
            k2=k2+i[1]**2
        k2=k2/G.number_of_nodes()
        print(k2/k**2)

def trainTime():#输出存放在ttt
    RE = []
    networks = ['BA1000_2', 'BA1000_5', 'BA1000_10',
              'BA3000_2', 'BA3000_5', 'BA3000_10'
              ]
    # networks = ['BA1000_2']
    L = [x for x in range(8, 48, 8)]
    for i in range(len(networks)):
        import time
        G = nx.read_edgelist('networks/' + networks[i])
        G.name = networks[i]
        Seconds = []
        for l in L:
            train_dataset = Dataset(G.name, l, 1.5)
            train_loader = DataLoader(train_dataset, batch_size=G.number_of_nodes(), shuffle=True)
            nodes = list(G.nodes())
            tt = time.time()
            PredictNodeByCNN(nodes, train_loader, l, 1.5, G.name, True)
            Seconds.append(time.time() - tt)
        RE.append(Seconds)
    print(RE)




if __name__ == '__main__':
    # G = nx.random_graphs.barabasi_albert_graph(3000, 10)
    # nx.write_edgelist(G, 'networks/BA3000_10')


    # K = [x for x in range(8, 48, 4)]
    # X = [x / 10.0 for x in range(10, 21)]
    # trains = ['BA1000_2', 'BA1000_5', 'BA1000_10'
    #           'BA3000_2', 'BA3000_5', 'BA3000_10'
    #           ]
    # networks = ['BA2000_2', 'BA2000_5', 'BA2000_10',
    #             'Jazz', 'Email', 'Oz',
    #             'Router', 'Faa', 'Facebook'
    #             ]
    # networks = ['BA4000_2', 'BA4000_5', 'BA4000_10',
    #             'BA8000_2', 'BA8000_5', 'BA8000_10',
    #             ]

    # draw_k(1.5)
    # heatmap()
    compareAll()
    # compareAll_topK(0.05)
    # networkFeat()
    # trainTime()

    # G = constructG('facebook')
    # print(degree(G))
    # print(betweenness(G))
    # print(VoteRank(G))
    # print(kShell(G))
    # G = nx.read_edgelist('networks/BA1000_2')
    # G.name = 'BA1000_2'
    # print(nx.is_connected(G))
    # print(G.number_of_nodes(), G.number_of_edges(), G.number_of_edges() * 2 / G.number_of_nodes())


    # for i in range(len(networks)):

        # G = nx.read_edgelist('networks/'+networks[i])
        # G.name = networks[i]
        # G = constructG(networks[i])
        # print(nx.is_connected(G))

        # k1 = 0.0
        # k2 = 0.0
        # for i in G.degree():
        #     k1 = k1 + i[1]
        #     k2 = k2 + i[1] ** 2
        # global UC
        # UC = k1 / (k2 - k1)
        #
        #
        # NeighborMatrix(G, 28)
        # print('矩阵生成成功')
        #
        # for x in X:
        #     label = []
        #     for node in G.nodes():
        #         label.append([SIR(G, [node], x, 1)])
        #     np.save('data_model_1/' + G.name + '_' + str(x) + '_label.npy', np.array(label))
        # print('标签生成成功')



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


