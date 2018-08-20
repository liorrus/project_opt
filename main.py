import numpy as np
import scipy as sc
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from math import ceil, floor
import threading
import asyncio
from random import uniform
from sklearn.model_selection import train_test_split
import multiprocessing

#return a dict of train, test features and results
def createPrevColumns(cols, numBack):
    columns = list(cols)
    for i in range(numBack):
        for col in cols:
            columns.append(col+str(i+1))
    del columns[columns.index('SHOT_RESULT')]
    return columns

def splitToTrainAndTest(df, test_size, cols, hmm, wf=False, n=3):
    columns = []
    if(hmm):
        if(wf):
            columns = createPrevColumns(cols+['SHOT_RESULT'], n)
        else:
            columns = cols+createPrevColumns(['SHOT_RESULT'], n)
    else:
        columns = cols
    return train_test_split(df, test_size=test_size)

def createFeaturesAndResults(data, columns):
    results = np.array(data['SHOT_RESULT'])
    features = data[columns]
    return {"features":features, "results":results}

def classifyAndGetConfDict(clf, trainDict, testDict):
    clf.fit(trainDict["features"], trainDict["results"])
    # print(sorted(list(zip(trainDict["features"], clf.feature_importances_)),key=lambda item: item[1],reverse=True))
    # print(clf.coef_)
    predictions = clf.predict(testDict["features"])
    return calculateAccuracy(predictions, testDict["results"])

def calculateAccuracy(predictions, results):
    true_p = 0
    false_p = 0
    true_n = 0
    false_n = 0
    size = len(predictions)
    for i in range(len(predictions)):
        if(predictions[i] == 1 and results[i] == 1):
            true_p += 1
        elif(predictions[i] == 0 and results[i] == 0):
            true_n += 1
        elif(predictions[i] == 1 and results[i] == 0):
            false_p += 1
        elif (predictions[i] == 0 and results[i] == 1):
            false_n += 1
    return {"tp":true_p/size,"tn":true_n/size,"fp":false_p/size,"fn":false_n/size}

def runRForestNTimes(n, trainDict, testDict):
    total = 0
    clf = RandomForestClassifier(n_estimators=1000)
    for i in range(n):
        confusionDict = classifyAndGetConfDict(clf, trainDict, testDict)
        total += confusionDict['tp']+confusionDict['tn']
    return total/n

def runAdvance(df, test_size, cols, hmm, wf=False, numBack=3, numIter=10, numExtend=100):
    train, test = splitToTrainAndTest(df, test_size, cols, hmm, wf, numBack)
    pre_extend_train, extended_train = createExtendedDataFrame(train, cols, numExtend)
    preExtendTrainDict = createFeaturesAndResults(pre_extend_train, cols)
    extendedTrainDict = createFeaturesAndResults(extended_train, cols)
    testDict = createFeaturesAndResults(test, cols)
    message = "ADVANCE: " + "Size: " + str(len(extended_train))+ " "
    if(hmm):
        message += str(hmm)+" NumBack:"+str(numBack)
        if(wf):
            message += " wf:"+str(wf)
    message += " res:"
    print("Pre Extend: ", message,runRForestNTimes(numIter, preExtendTrainDict, testDict))
    #print("Post Extend: ", message,runRForestNTimes(numIter, extendedTrainDict, testDict ))

def runBase(df, trainPercent, cols, hmm, wf=False, numBack=3, numIter=10):
    train, test = splitToTrainAndTest(df, trainPercent, cols, hmm, wf, n=numBack)
    trainDict = createFeaturesAndResults(train, cols)
    testDict = createFeaturesAndResults(test, cols)
    message = "BASE: "
    if(hmm):
        message += str(hmm)+" NumBack:"+str(numBack)
        if(wf):
            message += " wf:"+str(wf)
    message += " res:"
    print(message, runRForestNTimes(numIter, trainDict, testDict))

def createNewRecordsPerPlayer(numToCreate, cols, made_dict, missed_dict, diff_dict):
    made_item = {}
    missed_item = {}
    records = []
    frames = []
    for key in diff_dict.keys():
        for i in range(numToCreate):
            for col in cols:
                made_item[col] = made_dict[key][col] + uniform(-1*diff_dict[key][col], diff_dict[key][col])
                missed_item[col] = missed_dict[key][col] + uniform(-1*diff_dict[key][col], diff_dict[key][col])
            made_item['SHOT_RESULT'] = 1
            missed_item['SHOT_RESULT'] = 0
            records.append(made_item)
            records.append(missed_item)
            made_item = {}
            missed_item = {}
        frames.append(pd.DataFrame(records))
        records = []
    return pd.concat(frames)

def createExtendedDataFrame(df, cols, numPerPlayer):
    df2 = df[cols+['SHOT_RESULT', 'player_id']]
    df_by_player = df2.groupby('player_id')
    # keys are palyer ids and values are dicts containing made_data and missed_data
    sample_player_data = {}
    # key is player_id and value is dict containing for each column the medain of the column
    sample_made_mean = {}
    sample_missed_mean = {}
    player_mean_diff = {}

    for key, data in df_by_player:
        sample_player_data[key] = {}
        tmp = data[data['SHOT_RESULT']==1].sample(5)
        sample_player_data[key]['made'] = tmp[cols+['SHOT_RESULT']]
        tmp = data[data['SHOT_RESULT']==0].sample(5)
        sample_player_data[key]['missed'] = tmp[cols+['SHOT_RESULT']]

    for key in sample_player_data.keys():
        sample_made_mean[key] = {}
        sample_missed_mean[key] = {}
        player_mean_diff[key] = {}
        for col in cols:
            sample_made_mean[key][col] = sample_player_data[key]['made'][col].mean()
            sample_missed_mean[key][col] = sample_player_data[key]['missed'][col].mean()
            player_mean_diff[key][col] = abs(sample_made_mean[key][col] - sample_missed_mean[key][col])/2
    
    frames = []
    for data in sample_player_data.values():
        frames.append(pd.concat([data['made'], data['missed']]))
    tmp_df = pd.concat(frames)
    return (tmp_df,pd.concat([tmp_df,createNewRecordsPerPlayer(numPerPlayer, cols, sample_made_mean, sample_missed_mean, player_mean_diff)]))

if __name__=="__main__":
    df_temp = pd.read_csv('dfifull_no_nan.csv')
    df = df_temp.query('player_id == player_id3')
    cols = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'SHOT_CLOCK']

    # pool1 = []
    # print("Running NO HMM base and advance...")
    # # testing base no hmm
    # p = multiprocessing.Process(target=runBase, args=(df, 0.2, cols, False))
    # pool1.append(p)
    # p.start()
    # p = multiprocessing.Process(target=runAdvance, args=(df, 0.2, cols, False,))
    # pool1.append(p)
    # p.start()
    # for i in range(2):
    #     pool1[i].join()
    
    # print("Running Base HMM no features...")
    # pool2 = []
    # # testing base hmm without features
    # for i in range(1,4):
    #     p = multiprocessing.Process(target=runBase, args=(df, 0.2, cols, True, False, i,))
    #     pool2.append(p)
    #     p.start()
    # for i in range(3):
    #     pool2[i].join()
    
    # print("Running Base HMM with features...")
    # pool3 = []
    # # testing base hmm with features
    # for i in range(1,4):
    #     p = multiprocessing.Process(target=runBase, args=(df, 0.2, cols, True, True, i,))
    #     pool3.append(p)
    #     p.start() 
    # for i in range(3):
    #     pool3[i].join()

    print("Running Advance HMM no features...")
    pool4 = []
    # #testing advance hmm without features
    for i in range(1,4):
        p = multiprocessing.Process(target=runAdvance, args=(df, 0.2, cols, True, False, i,10,500))
        pool4.append(p)
        p.start()
    for i in range(3):
        pool4[i].join()

    print("Running Advance HMM with features...")
    # #testing advance hmm with features
    for i in range(1,4):
        p = multiprocessing.Process(target=runAdvance, args=(df, 0.2, cols, True, True, i,10,500))
        p.start()