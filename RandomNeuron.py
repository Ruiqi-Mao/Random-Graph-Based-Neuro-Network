import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy import linalg as LA

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1
    
def FourierApp(inputs,d_fea,kernel,bias,d):
    N = inputs.shape[0]
    for j in range(d):
        if j == 0:
            Zm = math.sqrt(1/d)*math.sqrt(2)*np.cos(np.dot(inputs,np.reshape(kernel[:,j],(d_fea,1)))+ bias[0][j])
        else:
            Z = math.sqrt(1/d)*math.sqrt(2)*np.cos(np.dot(inputs,np.reshape(kernel[:,j],(d_fea,1)))+ bias[0][j])
            Zm = np.hstack((Zm,Z))
    return Zm

def Rewired_ER(N,p):
    adj = np.zeros([N,N])
    for i in range(N-1):
        for j in range(i+1,N):
            if (np.random.uniform(0,1)<p):
                adj[i][j] = 1
                adj[j][i] = 1

    for i in range(N-1):
        for j in range(i+1,N):
            ps = adj[i][j]
            if (ps == 1):
                adj[j][i] = 0
    return adj

def Weight_Gen_First(ER,d,M,sigma):
    n_ne = ER.shape[0]
    rudu = np.reshape(np.sum(ER,axis = 0),(1,n_ne))#按列求和
    n_ne = ER.shape[0]
    weight_list = []
    for i in range(n_ne):
        k = rudu[0][i]
        if (k == 0):
            kernel = sigma * np.random.randn(d*M,3,3,3) + 0
        elif (k == 1):
            kernel = sigma * np.random.randn(d*M,d*M,3,3) + 0
        else:
            k = int(k)
            kernel = sigma * np.random.randn(k*d*M,d*M,3,3) + 0
        weight_list.append(kernel)
    return weight_list

def Weight_Gen(d_fea,ER,d,M,sigma):
    n_ne = ER.shape[0]
    rudu = np.reshape(np.sum(ER,axis = 0),(1,n_ne))#按列求和
    n_ne = ER.shape[0]
    weight_list = []
    for i in range(n_ne):
        k = rudu[0][i]
        if (k == 0):
            kernel = sigma * np.random.randn(d_fea,d*M) + 0
        else:
            k = int(k)
            kernel = sigma * np.random.randn(d*M,k*d*M) + 0
        weight_list.append(kernel)
    return weight_list

def Bias_Gen(ER,d,M):
    n_ne = ER.shape[0]
    rudu = np.reshape(np.sum(ER,axis = 0),(1,n_ne))#按列求和
    n_ne = ER.shape[0]
    bias_list = []
    for i in range(n_ne):
        k = rudu[0][i]
        if (k == 0 or k == 1):
            bias = 2*3.1415*np.random.rand(1,d)
        else:
            k = int(k)
            bias = 2*3.1415*np.random.rand(1,k*d)
        bias_list.append(bias)
    return bias_list
def find_zero(x):#找出初始节点的下标索引值
    row = x.shape[0]
    col = x.shape[1]
    zlist = []
    for i in range(row):
        for j in range(col):
            if (x[i][j] == 0):
                zlist.append(j)
    changdu = len(zlist)
    return zlist, changdu
def find_one(x):#找出1的下标索引值
    row = x.shape[0]
    col = x.shape[1]
    zlist = []
    for i in range(row):
        for j in range(col):
            if (x[i][j] == 1):
                zlist.append(i)
    changdu = len(zlist)
    return zlist, changdu

def del_same_elements(num1,num2):
    for i in range(0,len(num1)):
        for j in range(0,len(num2)):
            if num1[i] == num2[j]:
                same = num1[i]
            else:
                same = None
    num1.remove(same)
    #num2.remove(same)
    return num1

def ismember(A, B):
    nA = len(A)
    nB = len(B)
    A = np.array(A)
    B = np.array(B)
    A = np.reshape(A,(nA,1))
    B = np.reshape(B,(nB,1))
    return [ np.sum(a == B) for a in A ]

def BuildNeuron(inputs,G_f,weight,bias,Wfs,d,M,N3,N4,s):#N:输入维度,Batch：batch_size,
    n_ne = G_f.shape[0]#n_ne代表随机图邻接矩阵维数，也就是神经元个数
    rudu = np.reshape(np.sum(G_f,axis = 0),(1,n_ne))#按列求和
    initial_node, size_initial = find_zero(rudu)
    eS_list = []
    st = []
    N1 = inputs.shape[0]
    N2 = inputs.shape[1]
##构造神经元
    FF1 = np.zeros([N1,n_ne*d*M])

    distOfMaxAndMin = np.zeros([n_ne,1])
    minOfEachWindow = np.zeros([n_ne,1])
    for i in range(size_initial):
        k = initial_node[i]
        eS_list.append(k)
        Zm = np.zeros([N1,d*M])
        we = weight[k]
        bia = bias[k]
        for km in range(M):
            Z = FourierApp(inputs,N2,we[:,km*d:(km+1)*d],bia,d)
            Zm[:,km*d:(km+1)*d] = Z
        W = Wfs[k]
        Zm = np.dot(Zm,W)
        print('Neuron in random graph (initial neurons): max:',np.max(Zm),'min:',np.min(Zm))
        distOfMaxAndMin[k][0] = np.max(Zm) - np.min(Zm)
        minOfEachWindow[k][0] = np.min(Zm)
        Zm = (Zm-minOfEachWindow[k][0])/distOfMaxAndMin[k][0]
        FF1[:,d*M*k:d*M*(k+1)] = Zm
    num_list = []
    for i in range(n_ne):
        num_list.append(i)
    for i in range(size_initial):
        num_list.remove(initial_node[i])
    while (len(eS_list)<n_ne):
        for i in range(len(num_list)):
            z = G_f[:,num_list[i]]
            z = np.reshape(z,(n_ne,1))
            if (np.sum(z) != 0):
                g, changduo = find_one(z)
                if all(ismember(g,eS_list)) == True:
                    st.append(num_list[i])
                    eS_list.append(num_list[i])
                    Zf = np.zeros([N1,d*M])
                    We = weight[num_list[i]]
                    Bi = bias[num_list[i]]
                    for m in range(len(g)):
                        vvk = FF1[:,d*M*g[m]:d*M*(g[m]+1)]
                        
                        we = We[:,m*M*d:(m+1)*M*d]
                        
                        bi = Bi[:,m*d:(m+1)*d]
                        Zm = np.zeros([N1,d*M])
                        for km in range(M):
                            wes = we[:,km*d:(km+1)*d]
                            Z = FourierApp(vvk,d*M,wes,bi,d)
                            Zm[:,km*d:(km+1)*d] = Z
                        Zf = Zf + Zm
                    W = Wfs[num_list[i]]
                    Zf = np.dot(Zf,W)
                    print('Neuron in random graph: max:',np.max(Zf),'min:',np.min(Zf))
                    distOfMaxAndMin[num_list[i]][0] = np.max(Zf) - np.min(Zf)
                    minOfEachWindow[num_list[i]][0] = np.min(Zf)
                    Zf = (Zf-minOfEachWindow[num_list[i]][0])/distOfMaxAndMin[num_list[i]][0]
                    
                    FF1[:,d*M*num_list[i]:d*M*(num_list[i]+1)] = Zf
        
        num_list = del_same_elements(num_list,st)
        st = []
##构造增强神经元
    H2 = np.hstack((FF1,0.1*np.ones([N1,1])))
    EE1 = np.zeros([N1,N3*N4])

    WhStore = []
    parameterOfShrinkStore = []
    for i in range(N4):
        if M*d*n_ne>=N3:
            wh = LA.orth(2 * np.random.randn(n_ne*M*d+1,N3)-1)
        else:
            wh = LA.orth(2 * np.random.randn(n_ne*M*d+1,N3).T-1).T
        tempOfOutputOfEnhanceLayer = np.dot(H2, wh)
        WhStore.append(wh)
        print('Enhance Neuron: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))
        parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
        parameterOfShrinkStore.append(parameterOfShrink)
        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
        EE1[:,N3*i:N3*(i+1)] = OutputOfEnhanceLayer
    NeuronGroup = np.hstack((FF1,EE1))
    return NeuronGroup, distOfMaxAndMin, minOfEachWindow, WhStore, parameterOfShrinkStore

def TestNeuron(inputs,G_f,weight,bias,Wfs,d,M,N3,N4,distOfMaxAndMin, minOfEachWindow,WhStore,parameterOfShrinkStore):#N:输入维度,Batch：batch_size,
    n_ne = G_f.shape[0]#n_ne代表随机图邻接矩阵维数，也就是神经元个数
    rudu = np.reshape(np.sum(G_f,axis = 0),(1,n_ne))#按列求和
    initial_node, size_initial = find_zero(rudu)
    eS_list = []
    st = []
    N1 = inputs.shape[0]
    N2 = inputs.shape[1]
##构造神经元
    FF1 = np.zeros([N1,n_ne*d*M])

    for i in range(size_initial):
        k = initial_node[i]
        eS_list.append(k)
        Zm =np.zeros([N1,d*M])
        we = weight[k]
        bia = bias[k]
        for km in range(M):
            Z = FourierApp(inputs,N2,we[:,km*d:(km+1)*d],bia,d)
            Zm[:,km*d:(km+1)*d] = Z
        W = Wfs[k]
        Zm = np.dot(Zm,W)
        print('Neuron in graph (initial neurons): max:',np.max(Zm),'min:',np.min(Zm))
        Zm = (Zm-minOfEachWindow[k][0])/distOfMaxAndMin[k][0]
        FF1[:,d*M*k:d*M*(k+1)] = Zm
    num_list = []
    for i in range(n_ne):
        num_list.append(i)
    for i in range(size_initial):
        num_list.remove(initial_node[i])
    while (len(eS_list)<n_ne):
        for i in range(len(num_list)):
            z = G_f[:,num_list[i]]
            z = np.reshape(z,(n_ne,1))
            if (np.sum(z) != 0):
                g, changduo = find_one(z)
                if all(ismember(g,eS_list)) == True:
                    st.append(num_list[i])
                    eS_list.append(num_list[i])
                    Zf = np.zeros([N1,d*M])
                    We = weight[num_list[i]]
                    Bi = bias[num_list[i]]
                    for m in range(len(g)):
                        vvk = FF1[:,d*M*g[m]:d*M*(g[m]+1)]
                        we = We[:,m*M*d:(m+1)*M*d]
                        bi = Bi[:,m*d:(m+1)*d]
                        Zm =np.zeros([N1,d*M])
                        for km in range(M):
                            wes = we[:,km*d:(km+1)*d]
                            Z = FourierApp(vvk,d*M,wes,bi,d)
                            Zm[:,km*d:(km+1)*d] = Z
                        Zf = Zf + Zm
                    W = Wfs[num_list[i]]
                    Zf = np.dot(Zf,W)
                    print('Neuron in random graph: max:',np.max(Zf),'min:',np.min(Zf))
                    Zf = (Zf-minOfEachWindow[num_list[i]][0])/distOfMaxAndMin[num_list[i]][0]
                    
                    FF1[:,d*M*num_list[i]:d*M*(num_list[i]+1)] = Zf
        
        num_list = del_same_elements(num_list,st)
        st = []
##构造增强神经元
    H2 = np.hstack((FF1,0.1*np.ones([N1,1])))
    EE1 = np.zeros([N1,N3*N4])

    for i in range(N4):
        wh = WhStore[i]
        tempOfOutputOfEnhanceLayer = np.dot(H2, wh) 
        print('Enhance Neuron: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))
        parameterOfShrink = parameterOfShrinkStore[i]
        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
        EE1[:,N3*i:N3*(i+1)] = OutputOfEnhanceLayer
    NeuronGroup = np.hstack((FF1,EE1))
    return NeuronGroup

