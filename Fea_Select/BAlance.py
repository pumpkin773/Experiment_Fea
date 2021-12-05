#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
# 使用imlbearn库中上采样方法中的SMOTE接口
import six
import sys
sys.modules['sklearn.externals.six'] = six
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade
from sklearn.linear_model import LogisticRegression


def SMOTE_balance(origin_data_frame):
    ## SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本
    # 读取训练数据
    #origin_data = pd.read_csv(srcFile)
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # print(x, y)
    # print(np.sum(y == 1), np.sum(y == 0))
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    x_smo, y_smo = smo.fit_sample(x, y)
    smote_data = pd.concat([pd.DataFrame(x_smo, columns=columns[:-1]), pd.DataFrame(y_smo, columns=['category'])], axis=1)
    return smote_data

def RandomOverSampler_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # 使用RandomOverSampler从少数类的样本中进行随机采样来增加新的样本使各个分类均衡
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_sample(x, y)
    resampled_data = pd.concat([pd.DataFrame(x_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
                           axis=1)
    return resampled_data


def ADASYN_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本
    x_adasyn, y_adasyn = ADASYN().fit_sample(x, y)
    adasyn_data = pd.concat([pd.DataFrame(x_adasyn, columns=columns[:-1]), pd.DataFrame(y_adasyn, columns=['category'])],
                           axis=1)
    return adasyn_data

def RandomUnderSampler_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据: 随机选取数据的子集.
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_sample(x, y)
    resampled_data = pd.concat(
        [pd.DataFrame(x_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
        axis=1)
    return resampled_data


def SMOTEENN_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # 在之前的SMOTE方法中, 当由边界的样本与其他样本进行过采样差值时, 很容易生成一些噪音数据.
    # 因此, 在过采样之后需要对样本进行清洗.
    # 这样TomekLink 与 EditedNearestNeighbours方法就能实现上述的要求.

    smote_enn = SMOTEENN(random_state=0)
    x_SMOTEENN, y_SMOTEENN = smote_enn.fit_sample(x, y)
    SMOTEENN_data = pd.concat(
        [pd.DataFrame(x_SMOTEENN, columns=columns[:-1]), pd.DataFrame(y_SMOTEENN, columns=['category'])],
        axis=1)
    return SMOTEENN_data

def SMOTETomek_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # 在之前的SMOTE方法中, 当由边界的样本与其他样本进行过采样差值时, 很容易生成一些噪音数据.
    # 因此, 在过采样之后需要对样本进行清洗.
    # 这样TomekLink 与 EditedNearestNeighbours方法就能实现上述的要求.

    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_sample(x, y)
    SMOTETomek_data = pd.concat(
        [pd.DataFrame(X_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
        axis=1)
    return SMOTETomek_data

def EasyEnsemble_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # # EasyEnsemble 通过对原始的数据集进行随机下采样实现对数据集进行集成.
    # EasyEnsemble 有两个很重要的参数: (i) n_subsets 控制的是子集的个数
    # and (ii) replacement 决定是有放回还是无放回的随机采样.
    ee = EasyEnsemble(random_state=0, n_subsets=10)
    X_resampled, y_resampled = ee.fit_sample(x, y)
    ee_data = pd.concat(
        [pd.DataFrame(X_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
        axis=1)
    return ee_data

def BalanceCascade_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # BalanceCascade(级联平衡)的方法通过使用分类器(estimator参数)来确保那些被错分类的样本
    # 在下一次进行子集选取的时候也能被采样到. 同样, n_max_subset 参数控制子集的个数,
    # 以及可以通过设置bootstrap=True来使用bootstraping(自助法).
    bc = BalanceCascade(random_state=0,
                        estimator=LogisticRegression(random_state=0),
                        n_max_subset=4)
    x_resampled, y_resampled = bc.fit_sample(x, y)
    bc_data = pd.concat(
        [pd.DataFrame(x_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
        axis=1)
    return bc_data


def BalancedBaggingClassifier_balance(origin_data_frame):
    columns = list(origin_data_frame.columns)
    x, y = origin_data_frame[columns[:-1]], origin_data_frame[columns[-1]]
    # BalanceCascade(级联平衡)的方法通过使用分类器(estimator参数)来确保那些被错分类的样本
    # 在下一次进行子集选取的时候也能被采样到. 同样, n_max_subset 参数控制子集的个数,
    # 以及可以通过设置bootstrap=True来使用bootstraping(自助法).
    bc = BalanceCascade(random_state=0,
                        estimator=LogisticRegression(random_state=0),
                        n_max_subset=4)
    x_resampled, y_resampled = bc.fit_sample(x, y)
    bc_data = pd.concat(
        [pd.DataFrame(x_resampled, columns=columns[:-1]), pd.DataFrame(y_resampled, columns=['category'])],
        axis=1)
    return bc_data

