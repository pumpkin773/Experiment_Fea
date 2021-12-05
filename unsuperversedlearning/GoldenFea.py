#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class GoldenFea:
    def __init__(self, ML_model):
        # 真实标签字典
        self.label_dic = {}
        # 处理好的dataframe，去除非int列，合并F-列
        self.origin_data_frame = None

        self.origin_fea = None
        if ML_model == 'SVC':
            self.model = SVC(kernel='linear', probability=True)
        elif ML_model == 'RF':
            self.model = RandomForestClassifier()
        elif ML_model == 'LR':
            self.model = LogisticRegression()
        self.evaluation_dic = {}

    def init_data(self, data_path):
        # 读取训练数据
        self.origin_data_frame = pd.read_csv(data_path)
        # 特征列
        columns = list(self.origin_data_frame.columns)
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

    def trim(self, df, common):
        header = df.columns
        for element in header:
            if element not in common:
                df = df.drop(element, axis=1)
        df_trim = df
        return df_trim

    def calculate_accuracy(self, seed = 0):
        np.random.seed(seed)
        common_header = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                      " F94", "F71", "F72", "F25", "F3", "F15", "F126", "F41", "F77"]
        common_header.append("category")

        self.origin_data_frame = self.trim(self.origin_data_frame, common_header)
        list_header = list(self.origin_data_frame.columns)
        kf = KFold(n_splits=10)
        accuracy, precision, recall, F1 = 0, 0, 0, 0
        ff = 0
        for train_index, test_index in kf.split(self.origin_data_frame):
            train = self.origin_data_frame.iloc[train_index]
            test = self.origin_data_frame.iloc[test_index]
            train_x, train_y = train[list_header[:-1]], list(train[list_header[-1]])
            test_x, test_y = test[list_header[:-1]], list(test[list_header[-1]])
            self.model.fit(train_x, train_y)
            predict = self.model.predict(test_x)
            #print(confusion_matrix(test_y, predict))
            #tn, fp, fn, tp = confusion_matrix(test_y, predict).ravel()

            tn, fp, fn, tp = np.fromiter((sum(
                bool(j >> 1) == bool(test_y[i]) and
                bool(j & 1) == bool(predict[i])
                for i in range(len(test_y))
            ) for j in range(4)), float)


            pre = tp / (tp + fp) if (tp + fp) != 0 else 0
            rec = tp / (tp + fn) if (tp + fn) != 0 else 0

            if pre == 0 or rec == 0:
                ff += 1
                continue
            accuracy += (tp + tn) / (tn + fp + fn + tp)
            precision += pre
            recall += rec
            F1 += 2 * pre * rec / (pre + rec) if (pre + rec) != 0 else 0
        if 10-ff == 0:
            self.evaluation_dic['accuracy'] = 0
            self.evaluation_dic['precision'] = 0
            self.evaluation_dic['recall'] = 0
            self.evaluation_dic['F1'] = 0
            return
        self.evaluation_dic['accuracy'] = accuracy / (10-ff)
        self.evaluation_dic['precision'] = precision / (10-ff)
        self.evaluation_dic['recall'] = recall / (10-ff)
        self.evaluation_dic['F1'] = F1 / (10-ff)
