import math
from sklearn.svm import SVC
import numpy as np
import pandas as pd

from unsuperversedlearning.Sample import Sample


class ExperimentTwo:
    def __init__(self):
        # 真实标签字典
        self.label_dic = {}
        # 处理好的dataframe，去除非int列，合并F-列
        self.origin_data_frame = None
        # 去除噪音特征的dataframe
        self.selected_data_frame = None
        self.model = SVC(kernel='linear', probability=True)
        # 样本集合，过滤噪声数据
        self.sample_list = []

    def init_data(self, data_path):
        # 读取训练数据
        self.origin_data_frame = pd.read_csv(data_path)

        # 特征列
        columns = list(self.origin_data_frame.columns)
        for index, row in self.origin_data_frame.iterrows():
            self.label_dic[index] = 1 if row['category'] == 'close' else 0
        # drop非数值列
        for col in columns:
            if not np.issubdtype(self.origin_data_frame[col], np.int64) and not np.issubdtype(
                    self.origin_data_frame[col], np.float64):
                del self.origin_data_frame[col]
        columns = list(self.origin_data_frame.columns)

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

    # 筛选特征
    def select_feature(self):
        columns = self.origin_data_frame.columns

        # 中值字典
        median_dic = {}
        for col in columns:
            median_dic[col] = np.median(self.origin_data_frame[col])

        # 记录每列的噪声数
        select_metric_dic = {key: 0 for key in columns}
        origin = self.origin_data_frame
        for index, row in self.origin_data_frame.iterrows():
            # 根据真实标签来记录违规度量数
            label = self.label_dic[index]
            for col in columns:
                if (label == 1 and row[col] <= median_dic[col]) or (label == 0 and row[col]
                                                                    >= median_dic[col]):
                    select_metric_dic[col] += 1
        stardard = min(select_metric_dic.values())
        for col in columns:
            if select_metric_dic[col] != stardard:
                # 删除噪声列
                del origin[col]
        self.selected_data_frame = origin

    # 筛选样本
    def select_instance(self):
        samples = []
        columns = self.selected_data_frame.columns
        # 中值字典
        median_dic = {}
        for col in columns:
            median_dic[col] = np.median(self.origin_data_frame[col])
        for index, row in self.origin_data_frame.iterrows():
            label = self.label_dic[index]
            flag = True
            feature = []
            for col in columns:
                feature.append(row[col])
                if (label == 1 and row[col] <= median_dic[col]) or (label == 0 and row[col]
                                                                    >= median_dic[col]):
                    # 筛选噪声数据
                    flag = False
                    break
            if flag:
                sample = Sample()
                sample.label = label
                sample.feature = feature
                samples.append(sample)
        self.sample_list = samples

    # 计算准确率
    def calculate_accuracy(self):
        samples = self.sample_list
        # 样本一分为二，前半部分作为测试集，后半部分作为训练集
        train_sample = [samples[index] for index in range(math.ceil(len(samples) / 2))]
        test_sample = [samples[index] for index in range(math.ceil(len(samples) / 2) , len(samples))]
        train_x = [samples[index].feature for index in range(len(train_sample))]
        train_y = [samples[index].label for index in range(len(train_sample))]

        test_x = [samples[index].feature for index in range(len(test_sample))]

        test_y = [samples[index].label for index in range(len(test_sample))]
        self.model.fit(train_x, train_y)
        result = self.model.predict(test_x)
        correct_sum = 0
        for index in range(len(result)):
            if result[index] == test_y[index]:
                correct_sum += 1
        print(correct_sum / len(test_y))

if __name__ == '__main__':
    e = ExperimentTwo()
    e.init_data('../data/totalFeatures1.csv')
    e.select_feature()
    e.select_instance()
    e.calculate_accuracy()