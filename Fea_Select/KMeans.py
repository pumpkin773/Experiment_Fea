#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import pandas as pd
import numpy as np
from pandas import read_csv
from scipy.stats import pearsonr

class Kmeans(object):

    def __init__(self, correlate, data_fea):
        self.correlate = correlate
        self.fea = data_fea

        # label = self.df[df.columns[-1]]
        # fea = self.df.iloc[:, :-1]
        fea = pd.DataFrame(self.fea.values.T, index=self.fea.columns, columns=self.fea.index)
        # fea每行是一个特征的所有值，每列是一个样本
        nodeNum = len(fea)
        #print(nodeNum)
        db_min = float('inf')
        # for k in range(2, 4):
        for k in range(2, int(np.sqrt(nodeNum) + 1)):
            # print("===============================")
            # print("k=="+str(k))
            clusterData, center = self.kmeans(fea, k)
            db = self.DB(k, fea, clusterData, center)
            if db<db_min:
                db_min = db
                self.k_optimal = k
                self.clusterData = clusterData[:,0]
                self.center = center

    def distance(self, vector1, vector2):
        if self.correlate=="Euclidean":
            return np.sqrt(sum((vector2 - vector1)**2))
        elif self.correlate=="Pearson":
            # 输出:(r, p)
            # r:相关系数[-1，1]之间
            # p:相关系数显著性
            t = pearsonr(vector1, vector2)
            return t[0]

    def initCentre(self, data, k):
        numSample, dim = data.shape
        centre = np.zeros((k,dim))
        for i in range(0, k):
            #print("now,the k=="+str(k))
            index = int(np.random.uniform(0, numSample))
            centre[i, :] = data[index, :]
        return centre

    def kmeans(self, data, k):
        data = np.array(data)
        numSample = data.shape[0]
        #print(numSample)
        clusterData = np.array(np.zeros((numSample,2)))
        clusterChanged = True
        center = self.initCentre(data, k)
        g=0
        while clusterChanged:
            clusterChanged = False
            for i in range(numSample):
                minDis = float('inf')
                minIndex = 0
                for j in range(k):
                    dis = self.distance(center[j,:], data[i,:])
                    if dis < minDis:
                        minDis = dis
                        minIndex = j
                        clusterData[i,1] = dis
                if clusterData[i, 0] != minIndex:
                    clusterData[i, 0] = minIndex
                    clusterChanged = True
            for j in range(k):
                cluster_index = np.nonzero(clusterData[:,0] == j)
                count = data[cluster_index]
                #print(center)
                #迭代
                center[j,:] = np.mean(count,axis=0)
                #center[j,:] = np.median(count,axis=0)
            g+=1
            #print(g)
        #print(clusterData, center)
        return clusterData, center

    def DB(self, k, data, clusterData, center):
        data = np.array(data)
        #类内平均离散度
        s = np.array(np.zeros(k))
        d = np.array(np.zeros((k,k)))
        r = np.array(np.zeros(k))
        db = 0
        for j in range(k):
            cluster_index = np.nonzero(clusterData[:,0] == j)
            # print("cluster_index=="+str(cluster_index))
            for i in cluster_index[0]:
                s[j] = s[j] + np.sqrt(sum((data[i] - center[j, :]) ** 2))
            s[j] /= len(cluster_index)
        # print("this if whinin categories dis")
        # print(s)
        for i in range(k):
            for j in range(k):
                d[i][j] = np.sqrt(sum((center[i, :] - center[j, :]) ** 2))
        # print("this if Between categories dis")
        # print(d)
        for i in range(k):
            r_max = -float('inf')
            for j in range(k):
                if j != i:
                    if (s[i]+s[j])/d[i][j] > r_max:
                        r[i] = (s[i] + s[j]) / d[i][j]
                        r_max = (s[i] + s[j]) / d[i][j]
            db += r[i]
        db /= k
        # print("this is DB value:"+str(db))
        return db