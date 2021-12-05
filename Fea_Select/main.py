#!/usr/bin/python
# -*- coding:utf-8 -*-

import glob

import os

import numpy as np
import pandas as pd
from pandas import read_csv
from time import *
from sklearn import preprocessing
import argparse
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from unsuperversedlearning.RealLabel import RealLabel
from unsuperversedlearning.rankData import rankData
from unsuperversedlearning.ExperimentOne import ExperimentOne
from unsuperversedlearning.ExperimentTwo import ExperimentTwo
import math
from unsuperversedlearning.GoldenFea import GoldenFea
from ClusterFeaSelect import *
from FeaSelect import *
#from BAlance import *
import time
from func_timeout import func_set_timeout, FunctionTimedOut


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='dir to open data files', default='../data/')
parser.add_argument('--result_dir', type=str, help='dir to open data files', default='../result/')
parser.add_argument('--num_run', type=int, help='Number of runs', default=10)
parser.add_argument('--balance_data', type=str, help="balance_dataset['NO','SMOTE',"
                                                     "'RandomOverSampler','ADASYN',"
                                                     "'RandomUnderSampler','SMOTEENN',"
                                                     "'SMOTETomek','EasyEnsemble',"
                                                     "'BalanceCascade','BalancedBaggingClassifier']",
                    default='NO')
'选择特征的方法'
parser.add_argument('--feaSelect', type=str, help='selection feature select method algorithm:'
                                                  '[DBSCAN, Variance, Correlate, Chi2, Muinfor, '
                                                  'recurElimination, Penalty_l1, Penalty_l1l2, '
                                                  'GBDT, RSM]',
                    default="RSM")
'k-means算法的相关性计算方式'
parser.add_argument('--correlate', type=str, help='Correlation algorithm:[Pearson, Euclidean]', default="Euclidean")

parser.add_argument('--eps', type=int, help='In DBSCAN 半径', default=11)
parser.add_argument('--min_Pts', type=int, help='In DBSCAN 圈住的点的个数/密度', default=1)

parser.add_argument('--k', type=int, help='Select fea number', default=9)
#parser.add_argument('--scheme', type=str, help='[scheme_4, scheme_5, scheme_6]', default="scheme_6")
parser.add_argument('--MLmodel', type=str, help="[SVC, RF, LR]['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']", default='NB')
'聚类后选择特征的方法'
parser.add_argument('--select_method', type=str, help='cluster(DBSCAN,k-means)——> feature select '
                                                      'method:[std, inclass_corr, LDF, entropy]',
                    default='LDF')
parser.add_argument('--LDF_r', type=int, help='select LDF method r', default=10)

'ant1数据集对应最优eps=9, min_Pts=1, LDF_r=2'
'ant5数据集对应最优eps=21, min_Pts=1, LDF_r=7'
'derby1数据集对应最优eps=34, min_Pts=1, LDF_r=12'
'derby5数据集对应最优eps=27, min_Pts=1, LDF_r=1'
'mvn1数据集对应最优eps=8, min_Pts=1, LDF_r=2'
'mvn5数据集对应最优eps=19, min_Pts=1, LDF_r=10'

'cass1数据集对应最优eps=3, min_Pts=1, LDF_r=3'
'cass5数据集对应最优eps=34, min_Pts=1, LDF_r=18'
'commoms1数据集对应最优eps=21, min_Pts=1, LDF_r=14'
'commoms5数据集对应最优eps=16, min_Pts=1, LDF_r=6'
'jmeter1数据集对应最优eps=20, min_Pts=1, LDF_r=6'
'jmeter5数据集对应最优eps=19, min_Pts=2, LDF_r=6'

'lucence5数据集对应最优eps=28, min_Pts=1, LDF_r=1'
'tomcat5数据集对应最优eps=27, min_Pts=1, LDF_r=16'
'phoenix5数据集对应最优eps=39, min_Pts=1, LDF_r=2'
'mvn5数据集对应最优eps=19, min_Pts=1, LDF_r=10'

args = parser.parse_args()
#print(args)
def data_preprocessing(data_frame):
    # 预处理
    values = data_frame.values  # dataframe转换为array
    values = values.astype('float32')  # 定义数据类型
    data = preprocessing.scale(values)
    zscore_fea = pd.DataFrame(data)  # 将array还原为dataframe
    zscore_fea.columns = data_frame.columns  # 命名标题行
    data_frame = zscore_fea
    return data_frame

def experiment_main():
    for srcFile in glob.glob(args.data_dir + '*.csv'):
        filename = srcFile.split('\\')[1]
        filename = filename.split('.')[0]
        filename = 'exper_' + filename
        evaluate = {}
        fea_accuracy, train_accuracy_1, train_accuracy_0, test_accuracy, accuracy, train_1, train_0, \
        test_precision,test_recall,test_F1,precision,recall,F1 = 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0
        for i in range(args.num_run):
            e1 = ExperimentOne(args.MLmodel)
            e1.init_data(srcFile)
            '预处理'
            e1.origin_data_frame = data_preprocessing(e1.origin_data_frame)

            if args.feaSelect == "DBSCAN":
                e1.origin_data_frame = cluster_select_fea(e1.origin_data_frame, args)
            elif args.feaSelect == "Variance":
                e1.origin_data_frame = variance_select_fea(e1.origin_data_frame, args.k)
            elif args.feaSelect == "Correlate":
                e1.origin_data_frame = correlate_select(e1.origin_data_frame, e1.label_dic, args.k)
            elif args.feaSelect == "Chi2":
                e1.origin_data_frame = chi2_select(e1.origin_data_frame, e1.label_dic, args.k)
            elif args.feaSelect == "Muinfor":
                e1.origin_data_frame = mutual_information(e1.origin_data_frame, e1.label_dic, args.k)
            elif args.feaSelect == "recurElimination":
                e1.origin_data_frame = recur_elimination(e1.origin_data_frame, e1.label_dic, args.k)
            elif args.feaSelect == "Penalty_l1":
                e1.origin_data_frame = Penalty_l1(e1.origin_data_frame, e1.label_dic)
            elif args.feaSelect == "Penalty_l1l2":
                e1.origin_data_frame = Penalty_l1l2(e1.origin_data_frame, e1.label_dic)
            elif args.feaSelect == "GBDT":
                e1.origin_data_frame = GBDT(e1.origin_data_frame, e1.label_dic)
            elif args.feaSelect == "RSM":
                e1.origin_data_frame = RSM(e1.origin_data_frame, e1.label_dic)

            '标记策略'
            if args.scheme == "scheme_4":
                e1.scheme_4()
            elif args.scheme == "scheme_5":
                e1.scheme_5()
            elif args.scheme == "scheme_6":
                e1.scheme_6()
            e1.select_instance()
            f = True
            for i in range(len(e1.train_sample)):
                if e1.train_sample[i].cluster != e1.train_sample[0].cluster:
                    f = False
                    break
            if f:
                return
            e1.calculate_accuracy()
            print(e1.evaluation_dic)
            fea_accuracy += e1.evaluation_dic['fea_accuracy']
            train_1 += e1.evaluation_dic['train_1']
            train_0 += e1.evaluation_dic['train_0']
            train_accuracy_1 += e1.evaluation_dic['train_accuracy_1']
            train_accuracy_0 += e1.evaluation_dic['train_accuracy_0']
            test_accuracy += e1.evaluation_dic['test_accuracy']
            test_precision += e1.evaluation_dic['test_precision']
            test_recall += e1.evaluation_dic['test_recall']
            test_F1 += e1.evaluation_dic['test_F1']
            accuracy += e1.evaluation_dic['accuracy']
            precision += e1.evaluation_dic['precision']
            recall += e1.evaluation_dic['recall']
            F1 += e1.evaluation_dic['F1']
        print("====this feature accuracy====")
        evaluate['feature_accuracy'] = fea_accuracy / args.num_run
        print(evaluate['feature_accuracy'])
        print("====this train_sample cluster==1 number====")
        evaluate['train_1_num'] = train_1 / args.num_run
        print(evaluate['train_1_num'])
        print("====this train_sample cluster==0 number====")
        evaluate['train_0_num'] = train_0 / args.num_run
        print(evaluate['train_0_num'])
        print("====this train_sample cluster==1 accuracy====")
        evaluate['train_1_accuracy'] = train_accuracy_1 / args.num_run
        print(evaluate['train_1_accuracy'])
        print("====this train_sample cluster==0 accuracy====")
        evaluate['train_0_accuracy'] = train_accuracy_0 / args.num_run
        print(evaluate['train_0_accuracy'])
        print("====this test data accuracy====")
        evaluate['test_accuracy'] = test_accuracy / args.num_run
        print(evaluate['test_accuracy'])
        print("====this test data precision====")
        evaluate['test_precision'] = test_precision / args.num_run
        print(evaluate['test_precision'])
        print("====this test data recall====")
        evaluate['test_recall'] = test_recall / args.num_run
        print(evaluate['test_recall'])
        print("====this test data F1====")
        evaluate['test_F1'] = test_F1 / args.num_run
        print(evaluate['test_F1'])
        print("====this all data accuracy====")
        evaluate['accuracy'] = accuracy / args.num_run
        print(evaluate['accuracy'])
        print("====this all data precision====")
        evaluate['precision'] = precision / args.num_run
        print(evaluate['precision'])
        print("====this all data recall====")
        evaluate['recall'] = recall / args.num_run
        print(evaluate['recall'])
        print("====this all data F1====")
        evaluate['F1'] = F1 / args.num_run
        print(evaluate['F1'])
        save(evaluate, filename)

def cal_evaluate(TN_sum,FP_sum,FN_sum,TP_sum,AUC_sum, evaluate):
    TN_sum /= args.num_run
    FP_sum /= args.num_run
    FN_sum /= args.num_run
    TP_sum /= args.num_run
    AUC_sum /= args.num_run
    evaluate['TN'] = TN_sum
    evaluate['FP'] = FP_sum
    evaluate['FN'] = FN_sum
    evaluate['TP'] = TP_sum
    evaluate['accuracy'] = (TN_sum + TP_sum) / (TN_sum + FP_sum + FN_sum + TP_sum)
    precision = TP_sum / (FP_sum + TP_sum) if (FP_sum + TP_sum) != 0 else 0
    evaluate['precision'] = precision
    recall = TP_sum / (FN_sum + TP_sum) if (FN_sum + TP_sum) != 0 else 0
    evaluate['recall'] = recall
    evaluate['F1'] = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    evaluate['AUC'] = AUC_sum
    PF = FP_sum / (FP_sum + TN_sum)
    F_MEASURE = 2 * recall * precision / (recall + precision)
    G_MEASURE = 2 * recall * (1 - PF) / (recall + 1 - PF)
    evaluate['F_MEASURE'] = F_MEASURE
    evaluate['G_MEASURE'] = G_MEASURE
    numerator = (TP_sum * TN_sum) - (FP_sum * FN_sum)  # 马修斯相关系数公式分子部分
    denominator = math.sqrt(
        (TP_sum + FP_sum) * (TP_sum + FN_sum) * (TN_sum + FP_sum) * (TN_sum + FN_sum))  # 马修斯相关系数公式分母部分
    MCC = numerator / denominator
    evaluate['MCC'] = MCC
    print("====this is accuracy====")
    print(evaluate['accuracy'])
    print("====this is precision====")
    print(evaluate['precision'])
    print("====this is recall====")
    print(evaluate['recall'])
    print("====this is F1====")
    print(evaluate['F1'])
    print("====this is AUC====")
    print(evaluate['AUC'])
    print("====this is F_MEASURE====")
    print(evaluate['F_MEASURE'])
    print("====this is G_MEASURE====")
    print(evaluate['G_MEASURE'])
    print("====this is MCC====")
    print(evaluate['MCC'])

@func_set_timeout(600)
def reallabel_Exper(srcFile, flag):
    # for srcFile in glob.glob(args.data_dir + '*.csv'):
    print(srcFile)
    filename = srcFile.split('\\')[1]
    #linux系统需要改成以下写法
    # filename = srcFile.split('/')[-1]
    filename = filename.split('.')[0]
    if filename == 'ant_totalFeatures5':
        args.eps = 21
        args.min_Pts = 2
        args.LDF_r = 7
    elif filename == 'derby_totalFeatures5':
        args.eps = 27
        args.LDF_r = 1
    elif filename == 'mvn_totalFeatures5':
        args.eps = 19
        args.LDF_r = 10
    # elif filename == 'cass_totalFeatures5':
    #     args.eps = 34
    #     args.LDF_r = 18
    elif filename == 'cass_totalFeatures5':
        args.eps = 43
        args.LDF_r = 16
    elif filename == 'commoms_totalFeatures5':
        args.eps = 16
        args.LDF_r = 6
    elif filename == 'jmeter_totalFeatures5':
        args.eps = 19
        args.min_Pts = 2
        args.LDF_r = 6
    # elif filename == 'lucence_totalFeatures5':
    #     args.eps = 28
    #     args.LDF_r = 1
    elif filename == 'lucence_totalFeatures5':
        args.eps = 41
        args.LDF_r = 1
    # elif filename == 'tomcat_totalFeatures5':
    #     args.eps = 27
    #     args.LDF_r = 16
    elif filename == 'tomcat_totalFeatures5':
        args.eps = 40
        args.LDF_r = 1
    # elif filename == 'phoenix_totalFeatures5':
    #     args.eps = 39
    #     args.LDF_r = 2
    elif filename == 'phoenix_totalFeatures5':
        args.eps = 47
        args.LDF_r = 12
    # elif filename == 'mvn_totalFeatures5':
    #     args.eps = 19
    #     args.LDF_r = 10
    filename = flag + filename
    print(filename)
    print(args)
    evaluate = {}
    time_dic = {}
    accuracy, precision, recall, F1 = 0, 0, 0, 0
    TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum = 0, 0, 0, 0, 0
    select_time_sum, totle_time_sum, predict_time_sum, fit_time_sum = 0, 0, 0, 0
    for i in range(args.num_run):
        begin_time = time.time()
        print("============begin_time=========")
        print(begin_time)
        r = RealLabel(args.MLmodel)
        r.init_data(srcFile)
        begin_select_time = time.time()
        print("============begin_select_time=========")
        print(begin_select_time)

        if args.feaSelect == "DBSCAN" or args.feaSelect == "k-means":
            r.origin_fea = cluster_select_fea(r.origin_fea, args)
        elif args.feaSelect == "Variance":
            r.origin_fea = variance_select_fea(r.origin_fea, args.k)
        elif args.feaSelect == "Correlate":
            r.origin_fea = correlate_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Chi2":
            r.origin_fea = chi2_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Muinfor":
            r.origin_fea = mutual_information(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "recurElimination":
            r.origin_fea = recur_elimination(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Penalty_l1":
            r.origin_fea = Penalty_l1(r.origin_fea, r.label_dic)
        elif args.feaSelect == "Penalty_l1l2":
            r.origin_fea = Penalty_l1l2(r.origin_fea, r.label_dic)
        elif args.feaSelect == "GBDT":
            r.origin_fea = GBDT(r.origin_fea, r.label_dic)
        elif args.feaSelect == "RSM":
            r.origin_fea = RSM(r.origin_fea, r.label_dic)
        end_select_time = time.time()
        print("============end_select_time=========")
        print(end_select_time)
        select_time = end_select_time-begin_select_time
        print("============select_time=========")
        print(select_time)
        select_time_sum += select_time
        # 选择特征后用真实标签训练
        r.origin_data_frame = pd.concat([r.origin_fea, r.origin_data_frame[r.origin_data_frame.columns[-1]]],
                                        axis=1)
        save_fea(r.origin_fea, 'Fea_'+filename)
        #['NO','SMOTE','RandomOverSampler','ADASYN','RandomUnderSampler',
        # 'SMOTEENN','SMOTETomek','EasyEnsemble','BalanceCascade','BalancedBaggingClassifier']
        '''
        if args.balance_data == 'SMOTE':
            SMOTE_balance(r.origin_data_frame)
        elif args.balance_data == 'RandomOverSampler':
            RandomOverSampler_balance(r.origin_data_frame)
        elif args.balance_data == 'ADASYN':
            ADASYN_balance(r.origin_data_frame)
        elif args.balance_data == 'RandomUnderSampler':
            RandomUnderSampler_balance(r.origin_data_frame)
        elif args.balance_data == 'SMOTEENN':
            SMOTEENN_balance(r.origin_data_frame)
        elif args.balance_data == 'SMOTETomek':
            SMOTETomek_balance(r.origin_data_frame)
        elif args.balance_data == 'EasyEnsemble':
            EasyEnsemble_balance(r.origin_data_frame)
        elif args.balance_data == 'BalanceCascade':
            BalanceCascade_balance(r.origin_data_frame)
        elif args.balance_data == 'BalancedBaggingClassifier':
            BalancedBaggingClassifier_balance(r.origin_data_frame)
        '''
        r.calculate_accuracy()
        end_time = time.time()
        print("============end_time=========")
        print(end_time)
        totle_time = end_time-begin_time
        totle_time_sum += totle_time
        print("============totle_time=========")
        print(totle_time)
        predict_time_sum += r.time_dic['predict_time']
        fit_time_sum += r.time_dic['fit_time']

        print("===="+str(i)+"========")
        print(r.evaluation_dic)
        TN_sum += r.evaluation_dic['TN']
        FP_sum += r.evaluation_dic['FP']
        FN_sum += r.evaluation_dic['FN']
        TP_sum += r.evaluation_dic['TP']
        AUC_sum += r.evaluation_dic['AUC']
        #col = list(r.origin_data_frame.columns)
    cal_evaluate(TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum, evaluate)
    #evaluate['select_metric'] = col
    time_dic['predict_time'] = predict_time_sum / args.num_run
    time_dic['fit_time'] = fit_time_sum / args.num_run
    time_dic['select_Fea_time'] = select_time_sum/args.num_run
    time_dic['totle_time'] = totle_time_sum / args.num_run

    save(time_dic, evaluate, filename)

@func_set_timeout(600)
def allFea_reallabel(srcFile, name):
    print(srcFile)
    #filename = srcFile.split('\\')[1]
    # linux系统需要改成以下写法
    filename = srcFile.split('/')[-1]
    print(filename)
    filename = filename.split('.')[0]
    filename = name + filename
    print(filename)
    evaluate = {}
    time_dic = {}
    accuracy, precision, recall, F1 = 0, 0, 0, 0
    TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum = 0, 0, 0, 0, 0
    totle_time_sum, predict_time_sum, fit_time_sum = 0, 0, 0
    for i in range(args.num_run):
        begin_time = time.time()
        r = RealLabel(args.MLmodel)
        r.init_data(srcFile)
        r.calculate_accuracy()
        end_time = time.time()
        totle_time = end_time - begin_time
        totle_time_sum += totle_time
        predict_time_sum += r.time_dic['predict_time']
        fit_time_sum += r.time_dic['fit_time']

        print("====" + str(i) + "========")
        print(r.evaluation_dic)
        TN_sum += r.evaluation_dic['TN']
        FP_sum += r.evaluation_dic['FP']
        FN_sum += r.evaluation_dic['FN']
        TP_sum += r.evaluation_dic['TP']
        AUC_sum += r.evaluation_dic['AUC']
    cal_evaluate(TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum, evaluate)
    time_dic['predict_time'] = predict_time_sum / args.num_run
    time_dic['fit_time'] = fit_time_sum / args.num_run
    time_dic['totle_time'] = totle_time_sum / args.num_run

    save(time_dic, evaluate, filename)

def GoldenFea_Exper():
    for srcFile in glob.glob(args.data_dir + '*.csv'):
        evaluate = {}
        accuracy, precision, recall, F1 = 0, 0, 0, 0
        ff=0
        for i in range(args.num_run):
            g = GoldenFea(args.MLmodel)
            g.init_data(srcFile)
            g.calculate_accuracy()
            print(g.evaluation_dic)
            if g.evaluation_dic['recall']==0 or g.evaluation_dic['precision'] ==0:
                ff+=1
                continue
            accuracy += g.evaluation_dic['accuracy']
            precision += g.evaluation_dic['precision']
            recall += g.evaluation_dic['recall']
            F1 += g.evaluation_dic['F1']

        print("====this is accuracy====")
        evaluate['accuracy'] = accuracy / (args.num_run-ff)
        print(evaluate['accuracy'])
        print("====this is precision====")
        evaluate['precision'] = precision / (args.num_run-ff)
        print(evaluate['precision'])
        print("====this is recall====")
        evaluate['recall'] = recall / (args.num_run-ff)
        print(evaluate['recall'])
        print("====this is F1====")
        evaluate['F1'] = F1 / (args.num_run-ff)
        print(evaluate['F1'])
        save(evaluate)

def save(time_dic, evaluate, filename) :
    'Save the results to a comma-separated file.'
    path = args.result_dir + filename +'.csv'
    print_header = not os.path.exists(path)
    dic_a = vars(args)
    with open(path, 'a') as f:
        if print_header:
            for k in dic_a.keys():
                print('"%s"' % k, end=',', file=f)
            for k in time_dic.keys():
                print('"%s"' % k, end=',', file=f)
            for k in evaluate.keys():
                print('"%s"' % k, end=',', file=f)
            print(file=f)

        for v in dic_a.values():
            print(v, end=',', file=f)
        for v in time_dic.values():
            print(v, end=',', file=f)
        for v in evaluate.values():
            print(v, end=',', file=f)
        print(file=f)

def save_fea(data_fea, filename):
    'Save the results to a comma-separated file.'
    path = args.result_dir + filename + '.csv'
    print_header = not os.path.exists(path)
    dic_a = vars(args)
    with open(path, 'a') as f:
        if print_header:
            for k in dic_a.keys():
                print('"%s"' % k, end=',', file=f)
            print(file=f)

        for v in dic_a.values():
            print(v, end=',', file=f)
        for v in data_fea.columns:
            print(v, end=',', file=f)
        print(file=f)

'聚类选择特征+真实标签'
def clu_select_real(name):
    feaSelect = ['DBSCAN']
    select_method = ['std', 'inclass_corr', 'LDF']
    MLmodel = ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
    # select_method = ['LDF']
    # MLmodel = ['LR']
    for srcFile in glob.glob(args.data_dir + '*.csv'):
        for fs in feaSelect:
            args.feaSelect = fs
            for ml in MLmodel:
                args.MLmodel = ml
                for sm in select_method:
                    args.select_method = sm

                    try:
                        reallabel_Exper(srcFile, name)
                    except FunctionTimedOut as e:
                        print('timeout', e)
                    '''
                    for e in range(40, 80):
                        args.eps = e
                        for m in range(1, 15):
                            args.min_Pts = m
                            if args.select_method == 'LDF':
                                for lr in range(1, 30):
                                    args.LDF_r = lr
                                    reallabel_Exper(srcFile, 'DBSCAN_select_')
                    '''




'常规选择特征+真实标签'
def tar_select_real(name):
    'Variance'
    feaSelect = ['Variance', 'Correlate', 'Chi2', 'Muinfor',
                 'recurElimination', 'Penalty_l1', 'Penalty_l1l2', 'GBDT']
    MLmodel = ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
    for srcFile in glob.glob(args.data_dir + '*.csv'):
        for fs in feaSelect:
            args.feaSelect = fs
            for ml in MLmodel:
                args.MLmodel = ml
                if args.feaSelect in ['Penalty_l1', 'Penalty_l1l2', 'GBDT']:
                    print(args)
                    try:
                        reallabel_Exper(srcFile, name)
                    except FunctionTimedOut as e:
                        print('timeout', e)
                else:
                    for k in range(12, 16):
                        args.k = k
                        print(args)
                        try:
                            reallabel_Exper(srcFile, name)
                        except FunctionTimedOut as e:
                            print('timeout', e)

'全部特征+真实标签'
def allFea_real(name):
    MLmodel = ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
    for srcFile in glob.glob(args.data_dir + '*.csv'):
        for ml in MLmodel:
            args.MLmodel = ml
            print(args)
            try:
                allFea_reallabel(srcFile, name)
            except FunctionTimedOut as e:
                print('timeout', e)

@func_set_timeout(600)
def reallabel_Exper_1(srcFile, flag):
    # for srcFile in glob.glob(args.data_dir + '*.csv'):
    print(srcFile)
    #filename = srcFile.split('\\')[1]
    #linux系统需要改成以下写法
    filename = srcFile.split('/')[-1]
    filename = filename.split('.')[0]
    filename = flag + filename
    print(filename)
    print(args)
    evaluate = {}
    time_dic = {}
    for i in range(args.num_run):
        begin_time = time.time()
        print("============begin_time=========")
        print(begin_time)
        r = RealLabel(args.MLmodel)
        r.init_data(srcFile)
        begin_select_time = time.time()
        print("============begin_select_time=========")
        print(begin_select_time)
        if args.feaSelect == "DBSCAN" or args.feaSelect == "k-means":
            r.origin_fea = cluster_select_fea(r.origin_fea, args)
        elif args.feaSelect == "Variance":
            r.origin_fea = variance_select_fea(r.origin_fea, args.k)
        elif args.feaSelect == "Correlate":
            r.origin_fea = correlate_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Chi2":
            r.origin_fea = chi2_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Muinfor":
            r.origin_fea = mutual_information(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "recurElimination":
            r.origin_fea = recur_elimination(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Penalty_l1":
            r.origin_fea = Penalty_l1(r.origin_fea, r.label_dic)
        elif args.feaSelect == "Penalty_l1l2":
            r.origin_fea = Penalty_l1l2(r.origin_fea, r.label_dic)
        elif args.feaSelect == "GBDT":
            r.origin_fea = GBDT(r.origin_fea, r.label_dic)
        end_select_time = time.time()
        print("============end_select_time=========")
        print(end_select_time)
        select_time = end_select_time-begin_select_time
        print("============select_time=========")
        print(select_time)
        # 选择特征后用真实标签训练
        r.origin_data_frame = pd.concat([r.origin_fea, r.origin_data_frame[r.origin_data_frame.columns[-1]]],
                                        axis=1)
        '''
        if args.balance_data == 'SMOTE':
            SMOTE_balance(r.origin_data_frame)
        elif args.balance_data == 'RandomOverSampler':
            RandomOverSampler_balance(r.origin_data_frame)
        elif args.balance_data == 'ADASYN':
            ADASYN_balance(r.origin_data_frame)
        elif args.balance_data == 'RandomUnderSampler':
            RandomUnderSampler_balance(r.origin_data_frame)
        elif args.balance_data == 'SMOTEENN':
            SMOTEENN_balance(r.origin_data_frame)
        elif args.balance_data == 'SMOTETomek':
            SMOTETomek_balance(r.origin_data_frame)
        elif args.balance_data == 'EasyEnsemble':
            EasyEnsemble_balance(r.origin_data_frame)
        elif args.balance_data == 'BalanceCascade':
            BalanceCascade_balance(r.origin_data_frame)
        elif args.balance_data == 'BalancedBaggingClassifier':
            BalancedBaggingClassifier_balance(r.origin_data_frame)
        '''
        r.calculate_accuracy()
        end_time = time.time()
        print("============end_time=========")
        print(end_time)
        totle_time = end_time-begin_time
        print("============totle_time=========")
        print(totle_time)
        predict_time = r.time_dic['predict_time']
        fit_time = r.time_dic['fit_time']

        print("===="+str(i)+"========")
        print(r.evaluation_dic)
        TN = r.evaluation_dic['TN']
        FP = r.evaluation_dic['FP']
        FN = r.evaluation_dic['FN']
        TP = r.evaluation_dic['TP']
        AUC = r.evaluation_dic['AUC']
        cal_evaluate(TN, FP, FN, TP, AUC, evaluate)
        time_dic['predict_time'] = predict_time
        time_dic['fit_time'] = fit_time
        time_dic['select_Fea_time'] = select_time
        time_dic['totle_time'] = totle_time
        save(time_dic, evaluate, filename)

@func_set_timeout(600)
def allFea_reallabel_1(srcFile, name):
    print(srcFile)
    #filename = srcFile.split('\\')[1]
    # linux系统需要改成以下写法
    filename = srcFile.split('/')[-1]
    print(filename)
    filename = filename.split('.')[0]
    filename = name + filename
    print(filename)
    evaluate = {}
    time_dic = {}
    for i in range(args.num_run):
        begin_time = time.time()
        r = RealLabel(args.MLmodel)
        r.init_data(srcFile)
        r.calculate_accuracy()
        end_time = time.time()
        totle_time = end_time - begin_time
        predict_time = r.time_dic['predict_time']
        fit_time = r.time_dic['fit_time']
        print("====" + str(i) + "========")
        print(r.evaluation_dic)
        TN = r.evaluation_dic['TN']
        FP = r.evaluation_dic['FP']
        FN = r.evaluation_dic['FN']
        TP = r.evaluation_dic['TP']
        AUC = r.evaluation_dic['AUC']
        cal_evaluate(TN, FP, FN, TP, AUC, evaluate)
        time_dic['predict_time'] = predict_time
        time_dic['fit_time'] = fit_time
        time_dic['totle_time'] = totle_time
        save(time_dic, evaluate, filename)


@func_set_timeout(600)
def rank_Exper(srcFile, flag):
    # for srcFile in glob.glob(args.data_dir + '*.csv'):
    print(srcFile)
    # filename = srcFile.split('\\')[1]
    #linux系统需要改成以下写法
    filename = srcFile.split('/')[-1]
    filename = filename.split('.')[0]
    print(filename)
    evaluate = {}
    TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum = 0, 0, 0, 0, 0
    for i in range(args.num_run):
        r = rankData(args.MLmodel)
        r.init_data(srcFile)
        if args.feaSelect == "DBSCAN" or args.feaSelect == "k-means":
            r.origin_fea = cluster_select_fea(r.origin_fea, args)
        elif args.feaSelect == "Variance":
            r.origin_fea = variance_select_fea(r.origin_fea, args.k)
        elif args.feaSelect == "Correlate":
            r.origin_fea = correlate_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Chi2":
            r.origin_fea = chi2_select(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Muinfor":
            r.origin_fea = mutual_information(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "recurElimination":
            r.origin_fea = recur_elimination(r.origin_fea, r.label_dic, args.k)
        elif args.feaSelect == "Penalty_l1":
            r.origin_fea = Penalty_l1(r.origin_fea, r.label_dic)
        elif args.feaSelect == "Penalty_l1l2":
            r.origin_fea = Penalty_l1l2(r.origin_fea, r.label_dic)
        elif args.feaSelect == "GBDT":
            r.origin_fea = GBDT(r.origin_fea, r.label_dic)
        # 选择特征后用真实标签训练
        r.origin_data_frame = pd.concat([r.origin_fea, r.origin_data_frame[r.origin_data_frame.columns[-1]]],
                                        axis=1)
        r.calculate_accuracy()
        #print(r.result_dataframe)
        print(i)
        filename1 = flag + str(i) + '_' + filename
        path1 = filename.split('_')[0] + filename.split('_')[1][-1]
        path = '../golden_select_result/' + path1 + '/' + filename1 + '.csv'
        r.result_dataframe.to_csv(path)
        print(r.evaluation_dic)
        TN_sum += r.evaluation_dic['TN']
        FP_sum += r.evaluation_dic['FP']
        FN_sum += r.evaluation_dic['FN']
        TP_sum += r.evaluation_dic['TP']
        AUC_sum += r.evaluation_dic['AUC']
        # col = list(r.origin_data_frame.columns)
    cal_evaluate(TN_sum, FP_sum, FN_sum, TP_sum, AUC_sum, evaluate)
    path_3 = '../golden_select_result/' + path1 + '/' + filename.split('_')[0] + '_F1.csv'
    print_header = not os.path.exists(path_3)
    dic_a = vars(args)
    with open(path_3, 'a') as f:
        if print_header:
            for k in dic_a.keys():
                print('"%s"' % k, end=',', file=f)
            for k in evaluate.keys():
                print('"%s"' % k, end=',', file=f)
            print(file=f)

        for v in dic_a.values():
            print(v, end=',', file=f)
        for v in evaluate.values():
            print(v, end=',', file=f)
        print(file=f)


def rankExp(name = 'rank'):
    feaSelect = ['recurElimination']
    # feaSelect = ['Variance']
    # ['NB', 'LR', 'SVM', 'RF', 'DT', 'KNN']
    # MLmodel = ['SVC', 'RF', 'LR']
    MLmodel = ['NB', 'SVM', 'DT', 'KNN', 'RF', 'LR']
    MLmodel = ['LR']
    args.feaSelect = 'recurElimination'
    for srcFile in glob.glob('../golden_data/*.csv'):
        for ml in MLmodel:
            args.MLmodel = ml
            for k in range(6, 91):
                args.k = k
                name1 = 'k=' + str(args.k) + '_' + args.MLmodel + '_' + name
                print(args)
                try:
                    rank_Exper(srcFile, name1)
                except FunctionTimedOut as e:
                    print('timeout', e)

@func_set_timeout(600)
def chooseFeaRank(srcFile, flag):
    # for srcFile in glob.glob(args.data_dir + '*.csv'):
    print(srcFile)
    # filename = srcFile.split('\\')[1]
    # linux系统需要改成以下写法
    filename = srcFile.split('/')[-1]
    filename = filename.split('.')[0]
    # print(filename)
    for i in range(args.num_run):
        r = rankData(args.MLmodel)
        r.init_data(srcFile)
        feaNum_dic = {'F34': 5, 'F103': 10, 'F105': 5, 'F109': 20, 'F113': 15, 'F116': 45, 'F117': 35, 'F118': 15, 'F119': 5,
         'F120': 10, 'F110': 20, 'F122': 5, 'F123': 5, 'F115': 45, 'F88': 10, 'F45': 15, 'F46': 10, 'F128': 5,
         'F134': 5, 'F136': 10, 'F137': 5, 'F6': 10, 'F13': 5, 'F5': 5, 'F102': 10, 'F104': 15, 'F112': 10, 'F121': 5,
         'F70': 5, 'F130': 10, 'F1': 15, 'F71': 5, 'F114': 25, 'F77': 5, 'F126': 5, 'F38': 10, 'F132': 10, 'F83': 10,
         'F74': 5, 'F84': 5, 'F146': 5, 'F4': 5, 'F66': 5, 'F69': 10, 'F68': 5, 'F95': 5, 'F79': 15, 'F43': 5, 'F42': 5,
         'F135': 5, 'F16': 5, 'F18': 5, 'F15': 5, 'F108': 10, 'F73': 5, 'F62': 5, 'F67': 5, 'F3': 5, 'F64': 5, 'F94': 5,
         'F22': 5, 'F111': 5, 'F61': 5, 'F40': 5, 'F127': 5, 'F131': 5, 'F39': 5}

        fea_dic = {k: v for k, v in feaNum_dic.items() if v >= 10}
        print(len(fea_dic))
        print(r.origin_data_frame)
        list_fea = []
        for k, v in fea_dic.items():
            list_fea.append(k)
        list_fea.append("category")
        print(list_fea)
        header = r.origin_data_frame.columns
        for element in header:
            if element not in list_fea:
                r.origin_data_frame = r.origin_data_frame.drop(element, axis=1)
        print(r.origin_data_frame)

        r.calculate_accuracy()
        # print(r.result_dataframe)
        print(i)
        path1 = '../golden_newFea_data/' + filename.split('_')[0] + '/'
        path2 = path1 + filename.split('_')[0] + '_' + flag + str(i) + '.csv'
        r.result_dataframe.to_csv(path2)

def chooseFeaRankMain(name = 'rank'):
    MLmodel = ['NB', 'SVM', 'DT', 'KNN', 'RF', 'LR']
    # MLmodel = ['LR']
    for srcFile in glob.glob('../golden_data/*.csv'):
        for ml in MLmodel:
            args.MLmodel = ml
            name1 = ml + '_' + name
            print(args)
            try:
                chooseFeaRank(srcFile, name1)
            except FunctionTimedOut as e:
                print('timeout', e)

if __name__ == '__main__':
    '自己的论文实验'
    #rankExp()
    # chooseFeaRankMain()
    '''
    r = RealLabel(args.MLmodel)
    r.init_data('../d/ant_totalFeatures5.csv')
    r.origin_data_frame.to_csv('../d/allfea.csv')
    '''

    args.balance_data = 'NO'
    # 'Correlate', 'Chi2', 'Muinfor', 'recurElimination', 'Penalty_l1', 'Penalty_l1l2', 'GBDT'
    feaSelect = ['Variance']
    MLmodel = ['LR', 'SVM', 'KNN']
    for srcFile in glob.glob('../data/*.csv'):
        for fs in feaSelect:
            args.feaSelect = fs
            for ml in MLmodel:
                args.MLmodel = ml
                if args.feaSelect in ['Penalty_l1', 'Penalty_l1l2', 'GBDT']:
                    print(args)
                    try:
                        reallabel_Exper(srcFile, 'general_selecte_')
                    except FunctionTimedOut as e:
                        print('timeout', e)
                else:
                    for k in range(12, 16):
                        args.k = k
                        print(args)
                        try:
                            reallabel_Exper(srcFile, 'general_selecte_')
                        except FunctionTimedOut as e:
                            print('timeout', e)






    '聚类选择特征+真实标签'
    # clu_select_real('DBSCAN_select_')
    '常规选择特征+真实标签'
    # tar_select_real('general_selecte_')
    '全部特征+真实标签'
    #allFea_real('allFea_')


    '''
    table = xlrd.open_workbook('../data/dataresult.xlsx')
    sheet = table.sheet_by_name("Sheet1")
    count_nrows = sheet.nrows  # 获取总行数
    for i in range(1,count_nrows):
        file = sheet.cell(i, 1).value
        if file == 'ant1':
            srcFile = '../d/ant_totalFeatures1.csv'
        elif file == 'ant5':
            srcFile = '../d/ant_totalFeatures5.csv'
        elif file == 'cass1':
            srcFile = '../d/cass_totalFeatures1.csv'
        elif file == 'cass5':
            srcFile = '../d/cass_totalFeatures5.csv'
        elif file == 'commons1':
            srcFile = '../d/commons_totalFeatures1.csv'
        elif file == 'commons5':
            srcFile = '../d/commons_totalFeatures5.csv'
        elif file == 'derby1':
            srcFile = '../d/derby_totalFeatures1.csv'
        elif file == 'derby5':
            srcFile = '../d/derby_totalFeatures5.csv'
        elif file == 'jmeter1':
            srcFile = '../d/jmeter_totalFeatures1.csv'
        elif file == 'jmeter5':
            srcFile = '../d/jmeter_totalFeatures5.csv'
        elif file == 'mvn1':
            srcFile = '../d/mvn_totalFeatures1.csv'
        elif file == 'mvn5':
            srcFile = '../d/mvn_totalFeatures5.csv'
        args.balance_data = sheet.cell(i, 2).value
        args.feaSelect = sheet.cell(i, 3).value
        args.eps = int(sheet.cell(i, 4).value)
        args.min_Pts = int(sheet.cell(i, 5).value)
        args.k = int(sheet.cell(i, 6).value)
        args.MLmodel = sheet.cell(i, 7).value
        args.select_method = sheet.cell(i, 8).value
        args.LDF_r = int(sheet.cell(i, 9).value)

        if sheet.cell(i, 2).value == 'NO':
            if sheet.cell(i, 4).value == -1:
                try:
                    allFea_reallabel_1(srcFile, 'allFea_')
                except FunctionTimedOut as e:
                    print('timeout', e)
            elif sheet.cell(i, 3).value == 'DBSCAN':
                try:
                    reallabel_Exper_1(srcFile, 'cluster_select_')
                except FunctionTimedOut as e:
                    print('timeout', e)
            else:
                try:
                    reallabel_Exper_1(srcFile, 'general_selecte_')
                except FunctionTimedOut as e:
                    print('timeout', e)
        else:
            try:
                reallabel_Exper_1(srcFile, 'banlance_cluster_select_')
            except FunctionTimedOut as e:
                print('timeout', e)
    '''