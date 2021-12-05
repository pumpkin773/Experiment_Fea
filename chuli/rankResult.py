#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


def save(evaluate, filename) :
    'Save the results to a comma-separated file.'
    path = filename
    print_header = not os.path.exists(path)
    with open(path, 'a') as f:
        if print_header:
            for k in evaluate.keys():
                print('"%s"' % k, end=',', file=f)
            print(file=f)

        for v in evaluate.values():
            print(v, end=',', file=f)
        print(file=f)

def rank_eva():
    evaluation_dic = {}
    for path1 in glob.glob('../golden_newFea_data/*'):
        for srcFile in glob.glob(path1 + '/*'):
            filename = srcFile.split('\\')[-1]
            # linux系统需要改成以下写法
            #filename = srcFile.split('/')[-1]
            filename = filename.split('_')
            if filename[1] == 'LR':
                print(filename)
                origin_data_frame = pd.read_csv(srcFile)
                evaluation_dic['ML'] = filename[1]
                evaluation_dic['rank'] = filename[2]
                df = origin_data_frame.sort_values(by="1", ascending=False)
                # print(df)
                i, sr, tp = 0, 0, 0
                tp_num = []
                for index, row in df.iterrows():
                    if row['category'] == 0:
                        i += 1
                    elif row['category'] == 1:
                        sr += i
                        tp += 1
                    tp_num.append(tp)
                avg = sr / tp
                evaluation_dic['rank_avg'] = avg
                # print(avg)
                outfile1 = '../golden_random_rank/' + filename[0] + '/' + filename[0] + "_LR_rank_result.csv"
                outfile2 = '../golden_random_rank/' + filename[0] + '/' + filename[0] + "_LR_tp_num_result.csv"
                save(evaluation_dic, outfile1)
                with open(outfile2, 'a') as f:
                    for v in tp_num:
                        print(v, end=',', file=f)
                    print(file=f)

'从rank_eva得到的数据求均值，将K和FPavg对应'
def rank_fea():
    fea_dic = {}
    for path1 in glob.glob('../goldenResult2/*'):
        for srcFile in glob.glob(path1 + '/*'):
            # print(srcFile)
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            # print(filename)
            if filename[1]=='rank':
                data_frame = pd.read_csv(srcFile)
                model_data = data_frame.groupby(['K_num'])['rank_avg'].mean()
                'Series变成dataframe,reset_index将index也变为数据'
                model_data = model_data.to_frame().reset_index()
                print(model_data)
                print(filename)
                outfile1 = '../goldenResult2/' + filename[0] + '/' + filename[0] + "_k+FPavg_result.csv"
                model_data.to_csv(outfile1)
                break

def selKnum():
    '对比找到最优的LR结果的选择的特征集的个数'
    feaNum_dic = {}
    # outfile = "../goldenResult2/fea_num.csv"
    for path1 in glob.glob('../goldenResult2/*'):
        for srcFile in glob.glob(path1 + '/*'):
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            if filename[-2] == 'k+FPavg':
                # print(filename)
                data_frame = pd.read_csv(srcFile)
                df = data_frame.sort_values(by="rank_avg", ascending=True)
                # print(df)
                feaNum_dic[filename[0]] = df.iat[0,1]
                print(feaNum_dic)
                break
    return feaNum_dic

'选择特征'
def chooseFea(feaNum_dic):
    '定义一个字典，每个特征出现一次+1，最后选择出现次数超过60%的'
    '我们9个数据集*跑5次=45*60%=27次'
    fea_dic = {}
    for path1 in glob.glob('../golden_select_result/*'):
        fea_dic_data = {}
        for srcFile in glob.glob(path1 + '/*'):
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            if filename[1] == 'LR':
                cfile = filename[0].split('=')
                if int(cfile[-1]) == int(feaNum_dic[filename[-2]]):
                    print(filename)
                    data_frame = pd.read_csv(srcFile)
                    fea = list(data_frame.columns)
                    for f in fea[1:-2]:
                        if not fea_dic.__contains__(f):
                            fea_dic[f] = 1
                        else:
                            fea_dic[f] += 1
                        if not fea_dic_data.__contains__(f):
                            fea_dic_data[f] = 1
                        else:
                            fea_dic_data[f] += 1
        print(fea_dic_data)
        print(len(fea_dic_data))
    print(fea_dic)
    print(len(fea_dic))
    print({k: v for k, v in fea_dic.items() if v >= 10})
    print(len({k: v for k, v in fea_dic.items() if v >= 10}))
    return {k: v for k, v in fea_dic.items() if v >= 10}

'取每个数据集中选择的特征，生成新的文件，然后去跑一下看看结果'
def new_csvdata(fea_dic):
    list_fea = []
    for k, v in fea_dic.items():
        list_fea.append(k)
    list_fea.append("category")
    print(list_fea)
    for srcFile in glob.glob('../golden_data/*.csv'):
        '未预处理，将他们的非数值去掉。。。'
        data_frame = pd.read_csv(srcFile)
        header = data_frame.columns
        for element in header:
            if element not in list_fea:
                data_frame = data_frame.drop(element, axis=1)
    print(data_frame)

def huatu():
    ml = 'LR'
    evaluation_dic = {}
    for path1 in glob.glob('../goldenResult2/*'):
        for srcFile in glob.glob(path1 + '/*'):
            # print(srcFile)
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            if filename[-2] == 'k+FPavg':
                print(filename)
                data_fram = pd.read_csv(srcFile)
                x = data_fram['K_num']
                y = data_fram['rank_avg']
                #color=['r','b','k','c','m','y','g','k','c','b']
                plt.plot(x, y, color = '#000079')  #, label=u'y=x^2曲线图'
                # plt.legend()  # 让图例生效
                # plt.margins(0)
                plt.subplots_adjust(bottom=0.15)
                plt.xlabel(u"Number of features")  # X轴标签
                plt.ylabel("FPavg")  # Y轴标签
                plt.title(filename[0])  # 标题
                plt.show()

def conRandomList():
    evaluation_dic = {}
    for path1 in glob.glob('../golden_select_result/*'):
        for srcFile in glob.glob(path1 + '/*'):
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            print(filename)
            if filename[-1] == 'totalFeatures5.csv':
                data_fram = pd.read_csv(srcFile)
                '打混后data_fram数据集的index（索引）还是按照正常的排序。'
                df = data_fram.sample(frac=1).reset_index(drop=True)
                # label_list = list(data_fram["category"])
                # print(label_list)
                i, sr, tp = 0, 0, 0
                tp_num = []
                for index, row in df.iterrows():
                    if row['category'] == 0:
                        i += 1
                    elif row['category'] == 1:
                        sr += i
                        tp += 1
                    tp_num.append(tp)
                avg = sr / tp
                evaluation_dic['ML'] = "random"
                evaluation_dic['rank_avg'] = avg
                # print(avg)
                outfile1 = '../golden_random_rank/' + filename[3] + '/' + filename[3] + "_rank_result.csv"
                outfile2 = '../golden_random_rank/' + filename[3] + '/' + filename[3] + "_tp_num_result.csv"
                save(evaluation_dic, outfile1)
                with open(outfile2, 'a') as f:
                    for v in tp_num:
                        print(v, end=',', file=f)
                    print(file=f)
                break

def culRandom():
    for path1 in glob.glob('../golden_random_rank/*'):
        random_list,LR_list = [],[]
        for srcFile in glob.glob(path1 + '/*'):
            # print(srcFile)
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            print(filename)
            if filename[-2]=='num':
                data_frame = pd.read_csv(srcFile, header=None)
                tp_list = data_frame.mean(axis=0)
                tp_rate = []
                print(tp_list)
                for i in range(len(tp_list) - 1):
                    tp_rate.append(tp_list[i] / (i + 1))
                if filename[1] == 'LR':
                    LR_list = tp_list
                else:
                    random_list = tp_list

        # color=['r','b','k','c','m','y','g','k','c','b']
        rand_list = list(random_list)
        rand_list = rand_list[0:-1]
        opt = [i for i in range(1, int(rand_list[-1])+1)]
        worst = [0 for i in range(1,len(rand_list)-int(rand_list[-1]))]
        opt_list = opt+[int(rand_list[-1]) for i in range(1,len(rand_list)-int(rand_list[-1]))]
        worst_list = worst+opt
        opt_x = range(len(opt_list))
        worst_x = range(len(worst_list))
        random_x = range(len(random_list))
        LR_x = range(len(LR_list))
        plt.plot(worst_x, worst_list, linestyle='--', color='#5B4B00',label='Worst')
        plt.plot(opt_x, opt_list, linestyle='--', color='#73BF00',label='Optimal')
        plt.plot(LR_x, LR_list, color='#000079',label='Our approach')
        plt.plot(random_x, random_list, color='#FFAF60',label='Random')
        plt.legend()  # 让图例生效
        # plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"Number of warnings")  # X轴标签
        plt.ylabel("Number of false positives")  # Y轴标签
        plt.title(filename[0])  # 标题
        plt.show()

def diffMLdata():
    evaluation_dic = {}
    for path1 in glob.glob('../golden_newFea_data/*'):
        for srcFile in glob.glob(path1 + '/*'):
            filename = srcFile.split('\\')[-1]
            # linux系统需要改成以下写法
            #filename = srcFile.split('/')[-1]
            filename = filename.split('_')
            print(filename)
            origin_data_frame = pd.read_csv(srcFile)
            evaluation_dic['ML'] = filename[1]
            evaluation_dic['rank'] = filename[2]
            df = origin_data_frame.sort_values(by="1", ascending=False)
            # print(df)
            i, sr, tp = 0, 0, 0
            tp_num = []
            for index, row in df.iterrows():
                if row['category'] == 0:
                    i += 1
                elif row['category'] == 1:
                    sr += i
                    tp += 1
                tp_num.append(tp)
            avg = sr / tp
            evaluation_dic['rank_avg'] = avg
            # print(avg)
            outfile1 = '../diff_ML_data/' + filename[0] + '/' + filename[0] + "_rank_result.csv"
            outfile2 = '../diff_ML_data/' + filename[0] + '/' + filename[0] + '_'+ filename[1]+"_tp_num_result.csv"
            save(evaluation_dic, outfile1)
            with open(outfile2, 'a') as f:
                for v in tp_num:
                    print(v, end=',', file=f)
                print(file=f)

def drawDiffML():
    for path1 in glob.glob('../diff_ML_data/*'):
        DT_list,KNN_list,LR_list,NB_list,RF_list,SVM_list = [],[],[],[],[],[]
        for srcFile in glob.glob(path1 + '/*'):
            # print(srcFile)
            filename = srcFile.split('\\')[-1]
            filename = filename.split('_')
            print(filename)
            if filename[2] != 'rank':
                data_frame = pd.read_csv(srcFile, header=None)
                tp_list = data_frame.mean(axis=0)
                tp_rate = []
                print(tp_list)
                for i in range(len(tp_list) - 1):
                    tp_rate.append(tp_list[i] / (i + 1))
                if filename[1] == 'LR':
                    LR_list = tp_list
                elif filename[1] == 'DT':
                    DT_list = tp_list
                elif filename[1] == 'KNN':
                    KNN_list = tp_list
                elif filename[1] == 'NB':
                    NB_list = tp_list
                elif filename[1] == 'RF':
                    RF_list = tp_list
                elif filename[1] == 'SVM':
                    SVM_list = tp_list

        DT_x = range(len(DT_list))
        KNN_x = range(len(KNN_list))
        LR_x = range(len(LR_list))
        NB_x = range(len(NB_list))
        RF_x = range(len(RF_list))
        SVM_x = range(len(SVM_list))
        plt.plot(DT_x, DT_list, linestyle='--', color='#5B4B00', label='DT')
        plt.plot(KNN_x, KNN_list, linestyle='--', color='#73BF00', label='KNN')
        plt.plot(LR_x, LR_list, linestyle='--', color='#000079', label='LR')
        plt.plot(NB_x, NB_list, linestyle='--', color='#FFAF60', label='NB')
        plt.plot(RF_x, RF_list, linestyle='--',color='#5A5AAD', label='RF')
        plt.plot(SVM_x, SVM_list, linestyle='--',color='#ff7575', label='SVM')
        plt.legend()  # 让图例生效
        # plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel(u"Number of warnings")  # X轴标签
        plt.ylabel("Number of false positives")  # Y轴标签
        plt.title(filename[0])  # 标题
        plt.show()

def showFea():
    filename1 = "../golden_newFea_data/ant/ant_DT_rank0.csv"
    fileMap = "../feature id mapping.csv"
    origin_data_frame1 = pd.read_csv(filename1)
    col = list(origin_data_frame1.columns)
    col = col[1:-2]
    print(len(col))
    origin_data_frame2 = pd.read_csv(fileMap)
    pro_list = list(origin_data_frame2['id in the program'])
    par_list = list(origin_data_frame2['id in the paper'])
    print(col)
    print(pro_list)
    print(par_list)
    weFea = []
    for c in col:
        c = c.split('F')[1]
        for ip in range(len(pro_list)):
            if np.isnan(pro_list[ip]):
                continue
            if int(c) == int(pro_list[ip]):
                print(int(c), pro_list[ip])
                weFea.append(par_list[ip])
    print(len(weFea))
    print(weFea)

if __name__ == '__main__':
    culRandom()
    # feaNum_dic = selKnum()
    # fea_dic = chooseFea(feaNum_dic)
    # new_csvdata(fea_dic)

    '''
    for i in range(10):
        conRandomList()
    '''
    # rank_eva()
    # culRandom()
    # huatu()
    # diffMLdata()
    # drawDiffML()



