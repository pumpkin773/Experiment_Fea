#!/usr/bin/python
# -*- coding:utf-8 -*-

import math

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


class RealLabel:
    def __init__(self, ML_model):
        # 真实标签字典
        self.label_dic = {}
        # 处理好的dataframe，去除非int列，合并F-列
        self.origin_data_frame = None

        self.origin_fea = None
        # ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
        if ML_model=='RF':
            self.model = RandomForestClassifier()
        elif ML_model=='NB':
            self.model = GaussianNB()
        elif ML_model=='LR':
            self.model = LogisticRegression()
        elif ML_model=='SVM':
            self.model = svm.SVC(kernel='linear', decision_function_shape='ovr')
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
        TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum = 0, 0, 0, 0, 0
        x, y = self.origin_data_frame[columns[:-1]], list(self.origin_data_frame[columns[-1]])
        for train_index, test_index in kf.split(x, y):
            train = self.origin_data_frame.iloc[train_index]
            test = self.origin_data_frame.iloc[test_index]
            train_x, train_y = train[columns[:-1]], list(train[columns[-1]])
            test_x, test_y = test[columns[:-1]], list(test[columns[-1]])
            begin_fit = time.time()
            self.model.fit(train_x, train_y)
            end_fit = time.time()
            fit_time = end_fit-begin_fit
            self.time_dic['fit_time'] = fit_time
            begin_predict = time.time()
            result = self.model.predict(test_x)
            end_predict = time.time()
            predict_time = end_predict - begin_predict
            self.time_dic['predict_time'] = predict_time
            TN, FP, FN, TP = np.fromiter((sum(
                bool(j >> 1) == bool(test_y[i]) and
                bool(j & 1) == bool(result[i])
                for i in range(len(test_y))
            ) for j in range(4)), float)
            AUC = roc_auc_score(test_y, result)
            TN_sum += TN
            FP_sum += FP
            FN_sum += FN
            TP_sum += TP
            AUC_sum += AUC
        TN_sum /= kf_num
        FP_sum /= kf_num
        FN_sum /= kf_num
        TP_sum /= kf_num
        AUC_sum /= kf_num
        self.evaluation_dic['TN'] = TN_sum
        self.evaluation_dic['FP'] = FP_sum
        self.evaluation_dic['FN'] = FN_sum
        self.evaluation_dic['TP'] = TP_sum
        self.evaluation_dic['accuracy'] = (TN_sum + TP_sum) / (TN_sum + FP_sum + FN_sum + TP_sum)
        precision = TP_sum / (FP_sum + TP_sum) if (FP_sum + TP_sum) != 0 else 0
        self.evaluation_dic['precision'] = precision
        recall = TP_sum / (FN_sum + TP_sum) if (FN_sum + TP_sum) != 0 else 0
        self.evaluation_dic['recall'] = recall
        self.evaluation_dic['F1'] = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        self.evaluation_dic['AUC'] = AUC_sum
        PF = FP_sum / (FP_sum + TN_sum)
        F_MEASURE = 2 * recall * precision / (recall + precision)
        G_MEASURE = 2 * recall * (1 - PF) / (recall + 1 - PF)
        self.evaluation_dic['F_MEASURE'] = F_MEASURE
        self.evaluation_dic['G_MEASURE'] = G_MEASURE
        numerator = (TP_sum * TN_sum) - (FP_sum * FN_sum)  # 马修斯相关系数公式分子部分
        denominator = math.sqrt((TP_sum + FP_sum) * (TP_sum + FN_sum) * (TN_sum + FP_sum) * (TN_sum + FN_sum))  # 马修斯相关系数公式分母部分
        MCC = numerator / denominator
        self.evaluation_dic['MCC'] = MCC

