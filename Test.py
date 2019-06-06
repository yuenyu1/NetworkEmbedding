import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import networkx as nx
import pandas as pd

def constructG(filename):
    G = nx.DiGraph().to_undirected()
    with open('networks/'+filename+'', 'r') as f:
        for position, line in enumerate(f):
            t = line.strip().split(' ')
            if t[0] != t[1]:
                G.add_edge(t[0], t[1])
    G.name = filename
    return G


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

    cnn = CNN(k)
    # optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

    # loss_fun
    loss_func = nn.MSELoss()

    # training loop
    if is_train:
        for epoch in range(100):
            for i, (x, y) in enumerate(data):
                batch_x = Variable(x)
                batch_y = Variable(y)
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

def corr(G, k, beta, train):
    label = np.load('data_model_1/'+G.name+'_'+str(beta)+'_label.npy')
    nodes = list(G.nodes())
    real = MakeReal(label, nodes)
    test_dataset = Dataset(G.name, k, beta)
    test_loader = DataLoader(test_dataset, batch_size=G.number_of_nodes(),  # 分批次训练
                             shuffle=False)
    rankCNN = PredictNodeByCNN(nodes,test_loader, k, beta, train, False)
    df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                       'CNN': np.array(nodesRank(rankCNN), dtype=float)
                       })
    return df.corr('kendall')['real']['CNN']

if __name__ == '__main__':
    G = nx.read_edgelist('networks/BA1000_2')
    G.name = 'BA1000_2'

    K = [x for x in range(8, 48, 4)]
    X = [x / 10.0 for x in range(10, 21)]


    # Train
    # for k in K:
    #     train_dataset = Dataset(G.name, k, 1.5)
    #     train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    #     nodes = list(G.nodes())
    #     PredictNodeByCNN(nodes, train_loader, k, 1.5, G.name, True)

    for k in K:
        print(corr(G, k, 1.5, 'BA1000_2'))