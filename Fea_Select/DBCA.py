#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict

class DBCA:

    def __init__(self, data_fea):
        #X, y = datasets.make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=[1.0, 1.0, 1.0],random_state=100)

        self.fea = data_fea
        fea = pd.DataFrame(self.fea.values.T, index=self.fea.columns, columns=self.fea.index)
        budget = 10
        distPercent = 2
        A, B = self.fit(fea, budget, distPercent)
        print("A+B:", len(A) + len(B))

        AA = fea[A]
        BB = fea[B]
        print(fea[A], fea[B])

        plt.scatter(AA[:, 0], AA[:, 1], marker='o')
        plt.scatter(BB[:, 0], BB[:, 1], marker='*')
        plt.show()

    def getDistCut(self, distList, distPercent):
        maxDist = max(distList)
        return maxDist * distPercent / 100

    def getRho(self, n, distMatrix, distCut):
        rho = np.zeros(n, dtype=float)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if distMatrix[i, j] < distCut:
                    rho[i] += 1
                    rho[j] += 1
        print("rho:", rho[:10])
        return rho

    def getGammaLeader(self, X, n, rho, distMatrix):
        Delta = np.zeros(n, dtype=float)
        Leader = np.ones(n, dtype=int) * (-1)
        OrdRhoIndex = np.flipud(np.argsort(rho))
        maxdist = 0
        for i in range(n):
            if distMatrix[OrdRhoIndex[0], i] > maxdist:
                maxdist = distMatrix[OrdRhoIndex[0], i]
        Delta[OrdRhoIndex[0]] = maxdist

        '''获取密度最大点以外样本的Delta和Leader'''
        for i in range(1, n):
            mindist = np.inf
            minindex = -1
            for j in range(i):
                if distMatrix[OrdRhoIndex[i], OrdRhoIndex[j]] < mindist:
                    mindist = distMatrix[OrdRhoIndex[i], OrdRhoIndex[j]]
                    minindex = OrdRhoIndex[j]
            Delta[OrdRhoIndex[i]] = mindist
            Leader[OrdRhoIndex[i]] = minindex
        Gamma = Delta * rho
        OrdGammaIndex = np.flipud(np.argsort(Gamma))
        print("Gamma", len(Gamma))
        # print(Gamma)
        print("OrdGammaIndex", len(OrdGammaIndex))
        print("Leader", len(Leader))
        print(Leader)
        EE = X[OrdGammaIndex[:3]]

        return Gamma, OrdGammaIndex, OrdRhoIndex, Leader


    def getInformationBlock(self, n, OrdGammaIndex, OrdRhoIndex, Leader):
        blockNum = 2
        clusterIndex = np.ones(n, dtype=int) * (-1)
        leftBlock = []
        rightBlock = []
        for j in range(blockNum):  ####直接给聚类中心点类簇标记{0,1}
            clusterIndex[OrdGammaIndex[j]] = j
        for i in range(1, n):
            if clusterIndex[OrdRhoIndex[i]] == -1:
                clusterIndex[OrdRhoIndex[i]] = clusterIndex[Leader[OrdRhoIndex[i]]]
        print("clusterIndex", set(clusterIndex))

        if len(set(clusterIndex)) != blockNum:
            print("密度峰值聚类环节出错了：类簇索引不是两个")
        for i in range(n):
            if clusterIndex[i] == 0:
                leftBlock.append(i)
            elif clusterIndex[i] == 1:
                rightBlock.append(i)
            else:
                print("出错了")
        return leftBlock, rightBlock  ####List类型


    def fit(self, X, budget, distPercent):
        n = len(X)
        distList = pdist(X, metric='cityblock')
        distMatrix = squareform(distList)
        distCut = self.getDistCut(distList, distPercent)
        rho = self.getRho(n, distMatrix, distCut)
        Gamma, OrdGammaIndex, OrdRhoIndex, Leader = self.getGammaLeader(X, n, rho, distMatrix)
        A, B = self.getInformationBlock(n, OrdGammaIndex, OrdRhoIndex, Leader)

        return A, B