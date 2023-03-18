import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import h5py
from RandomNeuron import Weight_Gen, Bias_Gen, BuildNeuron, TestNeuron, Rewired_ER
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.io as scio
 
def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z

def EMA_ADMM_L2_regulation(A, b, l2c, itrs):
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    bar = wk
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = (1*l2c + 1)**(-1)*(ck + uk)
        uk = uk + ck - ok
        wk = ok
        bar = (i+1)*bar/(i+3) + 2*wk/(i+3)
    return bar
def EMA_ADMM_L1_regulation(A, b, lam, itrs):
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    bar = wk
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
        bar = (i+1)*bar/(i+3) + 2*wk/(i+3)
    return bar
def show_accuracy(predictLabel, Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))
def pinv(A, reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)

if __name__ == "__main__":
##加载数据
    dataFile = './Fashion.mat'
    data = scio.loadmat(dataFile)
    train_x = np.double(data['train_x']/255)
    train_y = np.double(data['train_y'])
    test_x = np.double(data['test_x']/255)
    test_y = np.double(data['test_y'])

    Kmeans=KMeans(n_clusters=2,random_state=170) 
    Kmeans.fit(train_x)
    centers = Kmeans.cluster_centers_

    MinDistance = []
    for i in range(centers.shape[0]):
        DistanceForA = []
        for j in range(train_x.shape[0]):
            distance = np.linalg.norm(train_x[j,:] - centers[i,:],ord = 2)
            DistanceForA.append(distance)
        MinDistance.append(min(DistanceForA))
    sigma = (math.sqrt(min(MinDistance)))**(-1)
##稀疏编码
    N = train_x.shape[0]
    d_fea = train_x.shape[1]
    Nt = test_x.shape[0]
    df = 1500
    we = 2*np.random.randn(train_x.shape[1]+1,df) - 1
    H1 = np.hstack((train_x,0.1*np.ones((N,1))))
    H2 = np.hstack((test_x,0.1*np.ones((Nt,1))))
    H1 = np.vstack((H1,H2))
    A1 = np.dot(H1,we)
    scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(A1)
    A1AfterPreprocess = scaler1.transform(A1)
    SparseCodingWeight = EMA_ADMM_L1_regulation(A1AfterPreprocess,H1,0.001,100)
    SparseCodingWeight = SparseCodingWeight.T
    T1 = np.dot(H1,SparseCodingWeight)
    distOfMaxAndMinT = np.max(T1) - np.min(T1)
    minOfEachWindowT = np.min(T1)
    T1AfterPreprocess = (T1-minOfEachWindowT)/distOfMaxAndMinT
    train_x = T1AfterPreprocess[0:N,:]
    test_x = T1AfterPreprocess[N:N+Nt,:]
##构建第一个随机图
    NumberofNeuroninGraphOne = 16
    DimensionofFourierFeatureOne = 10
    WindowsperInputDegreeOne = 15
    NUmberofEnhancementNeuronsOne = 10
    DimensionofEnhancementNeuronsOne = 900
    ConnectionPaOne = 0.2
    ShrinkageCoffieOne = 0.8
    GraphOne = Rewired_ER(NumberofNeuroninGraphOne,ConnectionPaOne)

    FourierWeight = Weight_Gen(df,GraphOne,DimensionofFourierFeatureOne,WindowsperInputDegreeOne,sigma)
    FourierBias = Bias_Gen(GraphOne,DimensionofFourierFeatureOne,WindowsperInputDegreeOne)
    WeightofMappingNeuron = []
    for i in range(NumberofNeuroninGraphOne):
        WeightofMappingNeuron.append(2*np.random.rand(WindowsperInputDegreeOne*DimensionofFourierFeatureOne,WindowsperInputDegreeOne*DimensionofFourierFeatureOne)-1)

    NeuronGroupOne, distOfMaxAndMinOne, minOfEachWindowOne, WhStoreOne, parameterOfShrinkStoreOne = BuildNeuron(train_x,GraphOne,FourierWeight,FourierBias,WeightofMappingNeuron,DimensionofFourierFeatureOne,WindowsperInputDegreeOne,DimensionofEnhancementNeuronsOne,NUmberofEnhancementNeuronsOne,ShrinkageCoffieOne)
    #scalerNeuronGroupOne = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(NeuronGroupOne)
    #NeuronGroupOneAfterPreprocess = scalerNeuronGroupOne.transform(NeuronGroupOne)
    #NeuronGroupOne = NeuronGroupOneAfterPreprocess

##构建第二个随机图
    NumberofNeuroninGraphTwo = 16
    DimensionofFourierFeatureTwo = 10
    WindowsperInputDegreeTwo = 15
    NUmberofEnhancementNeuronsTwo = 10
    DimensionofEnhancementNeuronsTwo = 900
    ConnectionPaTwo = 0.2
    ShrinkageCoffieTwo = 0.8
    GraphTwo = Rewired_ER(NumberofNeuroninGraphTwo,ConnectionPaTwo)

    FourierWeightTwo = Weight_Gen(NeuronGroupOne.shape[1],GraphTwo,DimensionofFourierFeatureTwo,WindowsperInputDegreeTwo,sigma)
    FourierBiasTwo = Bias_Gen(GraphTwo,DimensionofFourierFeatureTwo,WindowsperInputDegreeTwo)
    WeightofMappingNeuronTwo = []
    for i in range(NumberofNeuroninGraphTwo):
        WeightofMappingNeuronTwo.append(2*np.random.rand(WindowsperInputDegreeTwo*DimensionofFourierFeatureTwo,WindowsperInputDegreeTwo*DimensionofFourierFeatureTwo)-1)

    NeuronGroupTwo, distOfMaxAndMinTwo, minOfEachWindowTwo, WhStoreTwo, parameterOfShrinkStoreTwo = BuildNeuron(NeuronGroupOne,GraphTwo,FourierWeightTwo,FourierBiasTwo,WeightofMappingNeuronTwo,DimensionofFourierFeatureTwo,WindowsperInputDegreeTwo,DimensionofEnhancementNeuronsTwo,NUmberofEnhancementNeuronsTwo,ShrinkageCoffieTwo)
    #scalerNeuronGroupTwo = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(NeuronGroupTwo)
    #NeuronGroupTwoAfterPreprocess = scalerNeuronGroupTwo.transform(NeuronGroupTwo)
    #NeuronGroupTwo = NeuronGroupTwoAfterPreprocess

##构建Pattern Matrix
    APatternMatrix = np.hstack((NeuronGroupOne,NeuronGroupTwo))
    #c = 2**-30
    #pinvOfInput = pinv(APatternMatrix,c)
    #OutputWeight = np.dot(pinvOfInput,train_y) 
    OutputWeight = EMA_ADMM_L2_regulation(APatternMatrix, train_y, 0.01, 200)
    OutputOfTrain = np.dot(APatternMatrix,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')

##测试阶段

    TestNeuronOne = TestNeuron(test_x,GraphOne,FourierWeight,FourierBias,WeightofMappingNeuron,DimensionofFourierFeatureOne,WindowsperInputDegreeOne,DimensionofEnhancementNeuronsOne,NUmberofEnhancementNeuronsOne,distOfMaxAndMinOne, minOfEachWindowOne, WhStoreOne, parameterOfShrinkStoreOne)
    #TestNeuronOneAfterPreprocess = scalerNeuronGroupOne.transform(TestNeuronOne)
    #TestNeuronOne = TestNeuronOneAfterPreprocess    

    TestNeuronTwo = TestNeuron(TestNeuronOne,GraphTwo,FourierWeightTwo,FourierBiasTwo,WeightofMappingNeuronTwo,DimensionofFourierFeatureTwo,WindowsperInputDegreeTwo,DimensionofEnhancementNeuronsTwo,NUmberofEnhancementNeuronsTwo,distOfMaxAndMinTwo, minOfEachWindowTwo, WhStoreTwo, parameterOfShrinkStoreTwo)
    #TestNeuronTwoAfterPreprocess = scalerNeuronGroupTwo.transform(TestNeuronTwo)
    #TestNeuronTwo = TestNeuronTwoAfterPreprocess
    APatternMatrixTest = np.hstack((TestNeuronOne,TestNeuronTwo))
    OutputOfTest = np.dot(APatternMatrixTest,OutputWeight)
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
