#!/usr/bin/python
# -*- coding:utf-8 -*-

import math
from random import Random

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import time
from numpy.random import permutation

from unsuperversedlearning.Sample import Sample


class rankData:
    def __init__(self, ML_model):
        # 真实标签字典
        self.label_dic = {}
        # 处理好的dataframe，去除非int列，合并F-列
        self.origin_data_frame = None

        self.origin_fea = None
        # ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
        if ML_model == 'RF':
            self.model = RandomForestClassifier()
        elif ML_model == 'NB':
            self.model = GaussianNB()
        elif ML_model == 'LR':
            self.model = LogisticRegression()
        elif ML_model == 'SVM':
            self.model = svm.SVC(kernel='linear', decision_function_shape='ovr', probability=True)
        elif ML_model == 'DT':
            self.model = DecisionTreeClassifier()
        elif ML_model == 'KNN':
            self.model = KNeighborsClassifier()
        self.evaluation_dic = {}
        self.time_dic = {}


    def init_data(self , data_path):
        # 读取训练数据
        self.origin_data_frame = pd.read_csv(data_path)
        # 特征列
        columns = list(self.origin_data_frame.columns)
        for index, row in self.origin_data_frame.iterrows():
            self.label_dic[index] = 1 if row['category'] == 'close' else 0
        self.origin_data_frame.loc[self.origin_data_frame['category'] == 'close', 'category'] = 1
        self.origin_data_frame.loc[self.origin_data_frame['category'] == 'open', 'category'] = 0
        label = self.origin_data_frame[self.origin_data_frame.columns[-1]]
        # drop非数值列
        for col in columns:
            if not np.issubdtype(self.origin_data_frame[col] , np.int64) and not np.issubdtype(self.origin_data_frame[col] , np.float64):
                del self.origin_data_frame[col]
        dic = {}
        for col in columns:
            key = col.split('-')[0]
            if not dic.__contains__(key):
                dic[key] = []
            dic[key].append(col)
        # 合并列
        for key, value in dic.items():
            if len(value) == 1:
                continue
            self.origin_data_frame[key] = np.zeros((len(self.origin_data_frame), 1))
            for col in value:
                self.origin_data_frame[key] += self.origin_data_frame[col]
                del self.origin_data_frame[col]
        self.origin_fea = self.origin_data_frame
        self.origin_fea = self.zscore_fea(self.origin_fea)
        self.origin_data_frame = pd.concat([self.origin_fea, label], axis=1)

    def zscore_fea(self, fea):
        # 预处理
        values = fea.values  # dataframe转换为array
        values = values.astype('float32')  # 定义数据类型
        data = preprocessing.scale(values)
        zscore_fea = pd.DataFrame(data)  # 将array还原为dataframe
        zscore_fea.columns = fea.columns  # 命名标题行
        return zscore_fea

    # 计算准确率
    def calculate_accuracy(self):
        columns = list(self.origin_data_frame.columns)
        kf_num = 10
        'StratifiedKFold保持数据样本比例'
        kf = StratifiedKFold(n_splits=kf_num)
        new_test_result_data, old_test_result_data = None, None
        x, y = self.origin_data_frame[columns[:-1]], list(self.origin_data_frame[columns[-1]])
        for train_index, test_index in kf.split(x, y):
            train = self.origin_data_frame.iloc[train_index]
            test = self.origin_data_frame.iloc[test_index]
            train_x, train_y = train[columns[:-1]], list(train[columns[-1]])
            test_x, test_y = test[columns[:-1]], test[columns[-1]]
            self.model.fit(train_x, train_y)
            result = self.model.predict(test_x)
            result_rank = self.model.predict_proba(test_x)
            result_01 = pd.DataFrame(result_rank, index=test_x.index, columns=[0, 1])
            old_test_result_data = pd.concat([test_x, test_y], axis=1)
            old_test_result_data = pd.concat([old_test_result_data, result_01[result_01.columns[-1]]], axis=1)
            if new_test_result_data is None:
                new_test_result_data = old_test_result_data
            else:
                new_test_result_data = pd.concat([new_test_result_data, old_test_result_data])
        self.result_dataframe = new_test_result_data



