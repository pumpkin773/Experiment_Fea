import math

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from numpy.random import permutation

from unsuperversedlearning.Sample import Sample


class ExperimentOne:
    def __init__(self, ML_model):
        # 真实标签字典
        self.label_dic = {}
        # 处理好的dataframe，去除非int列，合并F-列
        self.origin_data_frame = None
        # 样本集合
        self.sample_list = []
        # 训练样本集合
        self.train_sample = []
        # 测试样本集合
        self.test_sample = []
        if ML_model=='SVC':
            self.model = SVC(kernel='linear', probability=True)
        elif ML_model=='RF':
            self.model = RandomForestClassifier()
        elif ML_model=='LR':
            self.model = LogisticRegression()
        self.evaluation_dic = {}


    def init_data(self , data_path):
        # 读取训练数据
        self.origin_data_frame = pd.read_csv(data_path)

        # 特征列
        columns = list(self.origin_data_frame.columns)
        for index , row in self.origin_data_frame.iterrows():
            self.label_dic[index] = 1 if row['category'] == 'close' else 0
        # drop非数值列
        for col in columns:
            if not np.issubdtype(self.origin_data_frame[col] , np.int64) and not np.issubdtype(self.origin_data_frame[col] , np.float64):
                del self.origin_data_frame[col]
        columns = list(self.origin_data_frame.columns)

        dic = {}

        for col in columns:
            key = col.split('-')[0]
            if not dic.__contains__(key):
                dic[key] = []
            dic[key].append(col)
        # 合并列
        for key , value in dic.items():
            if len(value) == 1:
                continue
            self.origin_data_frame[key] = np.zeros((len(self.origin_data_frame) , 1 ))
            for col in value:
                self.origin_data_frame[key] += self.origin_data_frame[col]
                del self.origin_data_frame[col]
        '''
        # 预处理
        values = self.origin_data_frame.values  # dataframe转换为array
        values = values.astype('float32')  # 定义数据类型
        data = preprocessing.scale(values)
        zscore_fea = pd.DataFrame(data)  # 将array还原为dataframe
        zscore_fea.columns = self.origin_data_frame.columns  # 命名标题行
        self.origin_data_frame = zscore_fea
        '''

    def scheme_4(self):
        columns = self.origin_data_frame.columns
        # 中值字典
        median_dic = {}
        for col in columns:
            median_dic[col] = np.median(self.origin_data_frame[col])
        for index , row in self.origin_data_frame.iterrows():
            feature = [0 * len(columns)]
            s = Sample()
            for col in columns:
                digit = 1 if row[col] > median_dic[col] else 0
                s.weight += digit
                feature.append(digit)
            s.feature = feature
            s.label = self.label_dic[index]
            self.sample_list.append(s)
        # 标记
        self.label()
        # 计算准确率
        self.calculate_fea_accuracy()

    def scheme_5(self):
        columns = self.origin_data_frame.columns
        # 平均值字典
        mean_dic = {}
        for col in columns:
            mean_dic[col] = np.mean(self.origin_data_frame[col]) / 2
        for index, row in self.origin_data_frame.iterrows():
            feature = [0 * len(columns)]
            s = Sample()
            for col in columns:
                # 大于置位为1，小于为0
                digit = 1 if row[col] > mean_dic[col] else 0
                s.weight += digit
                feature.append(digit)
            s.feature = feature
            s.label = self.label_dic[index]
            self.sample_list.append(s)
        # 标记
        self.label()
        # 计算准确率
        self.calculate_fea_accuracy()

    # 标记样本
    def label(self):
        size = math.ceil(len(self.sample_list)/2)
        # 根据权重对样本进行排序
        self.sample_list = sorted(self.sample_list, key=lambda x: x.weight, reverse=True)
        for i in range(size):
            self.sample_list[i].cluster = 1
            cluster = self.sample_list[i].weight

        for i in range(size ,len(self.sample_list)):
            if cluster == self.sample_list[i].weight:
                self.sample_list[i].cluster = 1
            else:
                continue

    def calculate_fea_accuracy(self):
        accuracy_size = 0
        for i in range(len(self.sample_list)):
            sample = self.sample_list[i]
            if sample.label == sample.cluster:
                accuracy_size += 1
        self.evaluation_dic['fea_accuracy'] = accuracy_size / len(self.sample_list)
        print(accuracy_size / len(self.sample_list))

    def scheme_6(self):
        columns = self.origin_data_frame.columns
        # 将dataframe拆分为两半
        font_half_series = self.origin_data_frame.iloc[0:math.ceil(len(self.origin_data_frame) / 2)]
        back_half_series = self.origin_data_frame.iloc[math.ceil(len(self.origin_data_frame) / 2) + 1:]

        font_half_sum = 0
        back_half_sum = 0
        for col in columns:
            font_half_sum +=np.sum(font_half_series[col])
            back_half_sum +=np.sum(back_half_series[col])

        # 计算两个部分的平均值
        font_half_sum = font_half_sum / (math.ceil(len(self.origin_data_frame) / 2))
        back_half_sum = back_half_sum / math.ceil(len(self.origin_data_frame) / 2)

        # 正确标记的数目
        correct_sum = 0
        # 前半部分标记为1
        if font_half_sum / len(font_half_series) > back_half_sum / len(back_half_series):
            for index in range(math.ceil(len(self.origin_data_frame) / 2)):
                s = Sample()
                s.feature = self.origin_data_frame.iloc[index]
                s.label = self.label_dic[index]
                s.cluster = 1
                self.sample_list.append(s)
                if self.label_dic[index] == 1:
                    correct_sum += 1
            for index in range(math.ceil(len(self.origin_data_frame) / 2) , len(self.origin_data_frame)):
                s = Sample()
                s.feature = self.origin_data_frame.iloc[index]
                s.label = self.label_dic[index]
                s.cluster = 0
                self.sample_list.append(s)
                if self.label_dic[index] == 0:
                    correct_sum += 1
        # 后半部分标记为1
        else :
            for index in range(math.ceil(len(self.origin_data_frame) / 2)):
                s = Sample()
                s.feature = self.origin_data_frame.iloc[index]
                s.label = self.label_dic[index]
                s.cluster = 0
                self.sample_list.append(s)
                if self.label_dic[index] == 0:
                    correct_sum += 1
            for index in range(math.ceil(len(self.origin_data_frame) / 2) , len(self.origin_data_frame)):
                s = Sample()
                s.feature = self.origin_data_frame.iloc[index]
                s.label = self.label_dic[index]
                s.cluster = 1
                self.sample_list.append(s)
                if self.label_dic[index] == 1:
                    correct_sum += 1

        self.evaluation_dic['fea_accuracy'] = correct_sum / len(self.origin_data_frame)
        print(correct_sum / len(self.origin_data_frame))

    # 筛选样本
    def select_instance(self):
        # self.sample_list
        weight_1, weight_0 = self.sample_list[0].weight, self.sample_list[-1].weight
        for index in range(len(self.sample_list)):
            if self.sample_list[index].weight == weight_1:
                self.train_sample.append(self.sample_list[index])
            elif self.sample_list[index].weight == weight_0:
                self.train_sample.append(self.sample_list[index])
            else:
                self.test_sample.append(self.sample_list[index])

        '只有一个特征被选择时，很可能会全部划分为训练集'
        if len(self.test_sample)==0:
            print('All divided into training sets, reclassify')
            self.train_sample = []
            for i in range(0, math.ceil(0.05 * len(self.sample_list))):
                self.train_sample.append(self.sample_list[i])
            for i in range(math.ceil(0.05 * len(self.sample_list)), math.ceil(0.95 * len(self.sample_list))):
                self.test_sample.append(self.sample_list[i])
            for i in range(math.ceil(0.95 * len(self.sample_list)), len(self.sample_list)):
                self.train_sample.append(self.sample_list[i])

            # X_train, X_test, y_train, y_test = train_test_split(self.train_sample, self.train_sample, test_size=0.9, random_state=5)
        '''
        print("=====this is train_sample=========")
        for index in range(len(self.train_sample)):
           print(self.train_sample[index].weight, self.train_sample[index].cluster, self.train_sample[index].label)
        '''
        correct_sum_1, correct_sum_0, sum_1, sum_0 = 0, 0, 0, 0
        for index in range(len(self.train_sample)):
            if self.train_sample[index].cluster == 1:
                sum_1 += 1
                if self.train_sample[index].cluster == self.train_sample[index].label:
                    correct_sum_1 += 1
            else:
                sum_0 += 1
                if self.train_sample[index].cluster == self.train_sample[index].label:
                    correct_sum_0 += 1
        self.evaluation_dic['train_1'] = sum_1
        self.evaluation_dic['train_0'] = sum_0
        self.evaluation_dic['train_accuracy_1'] = correct_sum_1 / sum_1 if sum_1 != 0 else 0
        self.evaluation_dic['train_accuracy_0'] = correct_sum_0 / sum_0 if sum_0 != 0 else 0

    # 计算准确率
    def calculate_accuracy(self):
        train_x = [self.train_sample[index].feature for index in range(len(self.train_sample))]
        train_y = [self.train_sample[index].cluster for index in range(len(self.train_sample))]

        test_x = [self.test_sample[index].feature for index in range(len(self.test_sample))]
        test_y = [self.test_sample[index].label for index in range(len(self.test_sample))]

        self.model.fit(train_x, train_y)
        result = self.model.predict(test_x)

        test_TN, test_FP, test_FN, test_TP = np.fromiter((sum(
            bool(j >> 1) == bool(test_y[i]) and
            bool(j & 1) == bool(result[i])
            for i in range(len(test_y))
        ) for j in range(4)), float)
        #print(test_TN, test_FP, test_FN, test_TP)
        # 这是训练集的准确率
        test_accuracy = (test_TN + test_TP) / (test_TN + test_FP + test_FN + test_TP)
        test_precision = test_TP / (test_FP + test_TP) if (test_FP + test_TP) != 0 else 0
        test_recall = test_TP / (test_FN + test_TP) if (test_FN + test_TP) != 0 else 0
        test_F1 = 2*(test_precision*test_recall)/(test_precision+test_recall) if (test_precision+test_recall) != 0 else 0
        self.evaluation_dic['test_accuracy'] = test_accuracy
        self.evaluation_dic['test_precision'] = test_precision
        self.evaluation_dic['test_recall'] = test_recall
        self.evaluation_dic['test_F1'] = test_F1

        train_TN, train_FP, train_FN, train_TP = np.fromiter((sum(
            bool(j >> 1) == bool(self.train_sample[i].label) and
            bool(j & 1) == bool(train_y[i])
            for i in range(len(self.train_sample))
        ) for j in range(4)), float)
        #print(train_TN, train_FP, train_FN, train_TP)

        TN, FP, FN, TP = test_TN+train_TN, test_FP+train_FP, test_FN+train_FN, test_TP+train_TP
        accuracy = (TN + TP) / (TN + FP + FN + TP)
        precision = TP / (FP + TP) if (FP + TP) != 0 else 0
        recall = TP / (FN + TP) if (FN + TP) != 0 else 0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        self.evaluation_dic['accuracy'] = accuracy
        self.evaluation_dic['precision'] = precision
        self.evaluation_dic['recall'] = recall
        self.evaluation_dic['F1'] = F1


if __name__ == '__main__':
    e = ExperimentOne()
    e.init_data('../data/totalFeatures1.csv')
    e.scheme_6()