import numpy as np
import scipy as sc
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from math import ceil

#return a dict of train, test features and results
def createPrevColumns(cols):
    columns = list(cols)
    for i in range(1,3):
        for col in cols:
            columns.append(col+str(i))
    del columns[columns.index('SHOT_RESULT')]
    return columns

def createTrainAndTest(df, trainPercent, cols, hmm):
    size = len(df)
    trainSize = ceil(size*trainPercent)
    columns = []
    if(hmm):
        columns = createPrevColumns(cols)
    else:
        del cols[cols.index('SHOT_RESULT')]
        columns = cols
    train_sample = df[:trainSize]
    test_sample = df[trainSize:]
    train_results = np.array(train_sample['SHOT_RESULT'])
    train_features = train_sample[columns]
    test_results = np.array(test_sample['SHOT_RESULT'])
    test_features = test_sample[columns]
    return {"features":train_features, "results":train_results}, dict(features=test_features, results=test_results)

def classifyAndGetConfDict(clf, trainDict, testDict):
    clf.fit(trainDict["features"], trainDict["results"])
    # print(sorted(list(zip(trainDict["features"], clf.feature_importances_)),key=lambda item: item[1],reverse=True))
    # print(clf.coef_)
    predictions =  clf.predict(testDict["features"])
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
    clf = RandomForestClassifier(n_estimators=100)
    for i in range(n):
        confusionDict = classifyAndGetConfDict(clf, trainDict, testDict)
        total += confusionDict['tp']+confusionDict['tn']
    return total/n


if __name__=="__main__":
    #df = pd.read_csv('shot_log_train_factorized_no_nan.csv')
    df = pd.read_csv('dfi2_no_nan.csv')
    best = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'SHOT_CLOCK', 'SHOT_RESULT']
    best1 = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'SHOT_CLOCK', 'SHOT_RESULT']
    best2 = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'SHOT_CLOCK', 'SHOT_RESULT',
             'SHOT_RESULT1', 'SHOT_RESULT2']

    cols = ['LOCATION', 'SHOT_NUMBER', 'PERIOD',
            'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST',
            'CLOSE_DEF_DIST', 'SHOT_RESULT']
    cols1 = ['LOCATION', 'SHOT_NUMBER', 'PERIOD',
             'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST',
             'CLOSE_DEF_DIST', 'SHOT_RESULT']
    cols2 = ['LOCATION', 'SHOT_NUMBER', 'PERIOD',
             'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST',
             'CLOSE_DEF_DIST', 'SHOT_RESULT','SHOT_RESULT1','SHOT_RESULT2']
    #cols1=['SHOT_CLOCK', 'SHOT_NUMBER','DRIBBLES','TOUCH_TIME','SHOT_DIST','SHOT_RESULT','SHOT_RESULT1','SHOT_RESULT2']
    # df["missed_avg"] = df["missed_avg"] - 0.5478
    # df["made_avg"] = df["made_avg"] - 0.4086
    # df["TOUCH_TIME"] = df["TOUCH_TIME"] - 3.3092
    # df["CLOSE_DEF_DIST"] = df["CLOSE_DEF_DIST"] - 3.5869
    # df["SHOT_CLOCK"] = df["SHOT_CLOCK"] - 12
    # df["SHOT_DIST"] = df["SHOT_DIST"] - 14.23
    # df["W0"] = 1

    trainDict, testDict = createTrainAndTest(df, 0.8, best, False)
    trainDict1, testDict1 = createTrainAndTest(df, 0.8, best1, True)
    trainDict2, testDict2 = createTrainAndTest(df, 0.8, best2, False)
    print(runRForestNTimes(10, trainDict, testDict))
    print(runRForestNTimes(10, trainDict1, testDict1))
    print(runRForestNTimes(10, trainDict2, testDict2))
    clf = RandomForestClassifier(n_estimators=3000)
    # clf = Perceptron(max_iter=10000)
    # clf = SVC(kernel="linear")
    # confDict = classifyAndGetConfDict(clf, trainDict, testDict)
    # print(confDict["tp"]+confDict["tn"])
    # confDict = classifyAndGetConfDict(clf, trainDict1, testDict1)
    # print(confDict["tp"]+confDict["tn"])
    # confDict = classifyAndGetConfDict(clf, trainDict2, testDict2)
    # print(confDict["tp"]+confDict["tn"])
