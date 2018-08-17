import numpy as np
import scipy as sc
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from math import ceil, floor

#return a dict of train, test features and results
def createPrevColumns(cols, numBack):
    columns = list(cols)
    for i in range(numBack):
        for col in cols:
            columns.append(col+str(i+1))
    del columns[columns.index('SHOT_RESULT')]
    return columns

def createTrainAndTest(df, trainPercent, cols, hmm, wf = False, n=3):
    size = len(df)
    trainSize = ceil(size*trainPercent)
    columns = []
    if(hmm):
        if(wf):
            columns = createPrevColumns(cols+['SHOT_RESULT'], n)
        else:
            columns = cols+createPrevColumns(['SHOT_RESULT'], n)
    else:
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

def runAdvance(df, trainPercent, cols, hmm, wf=False, numBack=3, numIter=20):
    df_by_player = df.groupby('player_id')
    total_correct = 0
    total_test_size = 0
    for key, data in df_by_player:
        trainDict, testDict = createTrainAndTest(data, trainPercent, cols, hmm, wf, n=numBack)
        total_test_size += len(testDict["results"])
        total_correct += runRForestNTimes(numIter, trainDict, testDict)*len(testDict["results"])
    
    return total_correct/total_test_size

def runBase(df, trainPercent, cols, hmm, wf=False, numBack=3, numIter=20):
    trainDict, testDict = createTrainAndTest(df, trainPercent, cols, hmm, wf, n=numBack)
    return runRForestNTimes(numIter, trainDict, testDict)

if __name__=="__main__":
    df_temp = pd.read_csv('dfi1_no_nan.csv')
    df = df_temp.query('player_id == player_id2')
    cols = ['SHOT_DIST', 'TOUCH_TIME', 'CLOSE_DEF_DIST', 'SHOT_CLOCK']

    size = len(df)
    section_size = (len(df)/10)

    print("Running Base...")

    # testing base no hmm
    print(runBase(df, 0.8, cols, hmm=False))

    # testing base hmm without features
    for i in range(1,3):
        print(runBase(df, 0.8, cols, hmm=True, wf=False, numBack=i))
    
    # testing base hmm with features
    for i in range(1,3):
        print(runBase(df, 0.8, cols, hmm=True, wf=True, numBack=i))

    print("Running Advance...")
    
    #testing advance no hmm 
    print(runAdvance(df, 0.8, cols, hmm=False))

    #testing advance hmm without features
    for i in range(1,3):
        print(runAdvance(df, 0.8, cols, hmm=True, wf=False, numBack=i))

    #testing advance hmm with features
    for i in range(1,3):
        print(runAdvance(df, 0.8, cols, hmm=True, wf=True, numBack=i))
