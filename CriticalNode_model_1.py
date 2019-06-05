# -*- coding: utf-8 -*-
__author__ = '94353'

import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def Evaluate(G, rank, beta, r):
    infected = rank[:int(len(rank)*r)]
    return SIR(G, infected, beta, 1)

def constructG(filename):
    G = nx.DiGraph().to_undirected()
    with open('networks/'+filename+'', 'r') as f:
        for position, line in enumerate(f):
            t = line.strip().split(' ')
            if t[0] != t[1]:
                G.add_edge(t[0], t[1])
    G.name = filename
    return G



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
        Re = returnVect.reshape(k, k, 1)
        # returnVect = np.zeros((1, k*k))
        # for i in range(k):
        #     for j in range(k):
        #         returnVect[0, k * i + j] = subM[i, j]
        data.append(Re)
    np.save('data_model_1/' + G.name + '_'+str(k)+'_AM.npy', np.array(data))
    return data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 标准差为0.1的正态分布
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 偏差初始化为0.1
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def CNN(nodes, X_train, y_train, X_test, y_test, k, beta,is_train, name):
    x = tf.placeholder(tf.float32, shape=(None, k, k, 1))
    # x = tf.reshape(x, [-1, k, k, 1])
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    # -1代表先不考虑输入的图片例子多少这个维度，1是channel的数量
    keep_prob = tf.placeholder(tf.float32)

    # 构建卷积层1
    W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核5*5，1个channel，32个卷积核，形成32个featuremap
    b_conv1 = bias_variable([32])  # 32个featuremap的偏置
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)  # 用relu非线性处理
    h_pool1 = max_pool_2x2(h_conv1)  # pooling池化

    # 构建卷积层2
    W_conv2 = weight_variable([5, 5, 32, 64])  # 注意这里channel值是32
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 构建全连接层1
    W_fc1 = weight_variable([int(k/4 * k/4 * 64), 1024])
    b_fc1 = bias_variable([1024])
    h_pool3 = tf.reshape(h_pool2, [-1, int(k/4 * k/4 * 64)])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 构建全连接层2
    W_fc2 = weight_variable([1024, 1])
    b_fc2 = bias_variable([1])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    loss = tf.reduce_mean(tf.square(y_conv - y_train))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        if is_train:
            for i in range(2000):
                sess.run(train_step, feed_dict={x: X_train, y_: y_train, keep_prob: 0.5})
                if i % 100 == 0:
                    print("step %d, training accuracy %g" % (i, sess.run(loss, feed_dict={x: X_train, y_: y_train, keep_prob: 1})))
                try:
                    saver.save(sess, 'data_model_1/'+name+'_'+str(k)+'_'+str(beta)+'_ckpt/CN.ckpt', global_step=i + 1)
                except:
                    os.makedirs('data_model_1/'+name+'_'+str(k)+'_'+str(beta)+'_ckpt')
                    saver.save(sess, 'data_model_1/'+name + '_' + str(k) + '_' + str(beta) + '_ckpt/CN.ckpt', global_step=i + 1)
        else:
            model_file = tf.train.latest_checkpoint('data_model_1/'+name+'_'+str(k)+'_'+str(beta)+'_ckpt/')
            saver.restore(sess, model_file)
            predict = sess.run(y_conv, feed_dict={x: X_test, y_: y_test, keep_prob: 1})
            re = []
            for i in range(len(nodes)):
                re.append((nodes[i], predict[i]))
            rank = [x[0] for x in sorted(re, key=lambda x:x[1], reverse=True)]
            return rank
        sess.close()


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
    label = np.load('data/'+G.name+'_'+str(beta)+'_label.npy')
    nodes = list(G.nodes())
    real = MakeReal(label, nodes)
    rankCNN = CNN(nodes, np.load('data/'+train+'_'+str(k)+'_AM.npy'),
                  np.load('data/'+train+'_'+str(beta)+'_label.npy'),
                  np.load('data/'+G.name+'_'+str(k)+'_AM.npy'),
                  np.load('data/'+G.name+'_'+str(beta)+'_label.npy'), k, beta, False, train)
    df = pd.DataFrame({'real': np.array(nodesRank(real), dtype=float),
                       # 'degree': np.array(nodesRank(degree(G)), dtype=float),
                       # 'closeness': np.array(nodesRank(closeness(G)), dtype=float),
                       # 'betweenness': np.array(nodesRank(betweenness(G)), dtype=float),
                       # 'VR': np.array(nodesRank(VoteRank(G)), dtype=float),
                       # 'Kshell': np.array(nodesRank(kShell(G)), dtype=float),
                       'CNN': np.array(nodesRank(rankCNN), dtype=float)
                       })
    # return [df.corr('kendall')['real']['degree'],\
    #         df.corr('kendall')['real']['closeness'], \
    #         df.corr('kendall')['real']['betweenness'], \
    #         df.corr('kendall')['real']['VR'], \
    #         df.corr('kendall')['real']['Kshell'], \
    #         df.corr('kendall')['real']['CNN']]

    return df.corr('kendall')['real']['CNN']

# def draw(G, r):
#     N = float(G.number_of_nodes())
#     X = [float(y)/10.0 for y in range(16, 26)]
#     D = degree(G)
#     C = closeness(G)
#     K = kShell(G)
#     V = VoteRank(G)
#     HD = []
#     CL = []
#     Kshell = []
#     XGB = []
#     VR = []
#     for x in X:
#         print x
#         HD.append(Evaluate(G, D, x, r)/N)
#         CL.append(Evaluate(G, C, x, r)/N)
#         Kshell.append(Evaluate(G, K, x, r)/N)
#         VR.append(Evaluate(G, V, x, r) / N)
#         TrainData = np.load('data/Train_SIR.npy')
#         RXBG = Attack_XGB(TrainData, G, x)
#         XGB.append(Evaluate(G, RXBG, x, r)/N+0.01)
#         # tempData = TrainData[TrainData[:, -2] == x]
#         # tempArray = [str(int(y)) for y in tempData[tempData[:, -1].argsort()][::-1, 0]]
#         # POF.append(Evaluate(G, tempArray, x, r))
#
#     plt.plot(X, HD, 'g', label='HD', marker='.')
#     plt.plot(X, CL, 'y', label='CL', marker='*')
#     plt.plot(X, Kshell, 'b', label='Kshell', marker='>')
#     plt.plot(X, XGB, 'r', label='XGB', marker='^')
#     plt.plot(X, VR, 'y', label='VR', marker='<')
#     # plt.plot(X, XGB, 'r', label='XGB', marker='v')
#     plt.title(G.name)
#     plt.legend(loc='best', fontsize=14)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.xlabel(r'$\mu/\mu_c$', fontsize=30)
#     plt.ylabel(r'$R_s$', fontsize=30)
#     import matplotlib.ticker as ticker
#     plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#     plt.show()
#



if __name__ == '__main__':
    # G = constructG('powergrid')

    # G = nx.random_graphs.barabasi_albert_graph(1000, 2)
    # nx.write_edgelist(G, 'networks/BA1000_2')
    G = nx.read_edgelist('networks/BA1000_2')
    G.name = 'BA1000_2'
    k1 = 0.0
    k2 = 0.0
    for i in G.degree():
        k1 = k1 + i[1]
        k2 = k2 + i[1] ** 2
    global UC
    UC = k1 / (k2 - k1)

    K = [x for x in range(8, 48, 4)]
    X = [x/10.0 for x in range(10, 21)]

    # 生成矩阵
    # for k in K:
    #     NeighborMatrix(G, k)

    #生成标签
    # for x in [1.5]:
    #     label = []
    #     for node in G.nodes():
    #         label.append([SIR(G, [node], x, 1)])
    #     np.save('data_model_1/' + G.name + '_' + str(x) + '_label.npy', np.array(label))

    #Train
    for k in K:
        data = np.load('data_model_1/BA1000_2_'+str(k)+'_AM.npy')
        label = np.load('data_model_1/BA1000_2_1.5_label.npy')
        nodes = list(G.nodes())
        CNN(nodes, data, label, [], [], k, 1.5, True, G.name)

    # for k in K:
    #     print(corr(G, k, 1.5, 'BA500_5'))

    # data = np.load('data/BA500_5_' + str(32) + '_AM.npy')
    # label = np.load('data/BA500_5_1.5_label.npy')
    # data_test = np.load('data/Jazz_' + str(32) + '_AM.npy')
    # label_test = np.load('data/Jazz_1.5_label.npy')
    # nodes = list(G.nodes())
    # print(CNN(nodes, data, label, data_test, label_test, 32, 1.5, False, 'BA500_5'))
