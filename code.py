import numpy as np
import scipy as sc
import pandas as pd
import random
import sqlite3
import warnings
import csv
#from sqlalchemy import create_engine

def calculateAverage(df):
    return {
        "made": len(df[df["SHOT_RESULT"]=="made"]["SHOT_RESULT"])/len(df),
        "missed": len(df[df["SHOT_RESULT"]=="missed"]["SHOT_RESULT"])/len(df)
        }

def trivialClassifier(made_avg):
    rnd = random.random()
    return 'made' if (rnd <= made_avg) else 'missed'

def trivialClassifierExe(test, train):
    avgs = calculateAverage(train)
    true = 0
    for res in test['SHOT_RESULT']:
        if(res == trivialClassifier(avgs['made'])):
            true += 1
    return true/len(test)

def createDicts(off_id, def_id):
    off_dict = {}
    def_dict = {}
    for oid in off_id:
        # cell 1 is the number of made shots by the player,
        # cell 2 is the total number of shots by the player
        off_dict[oid] = [0.0,0.0]
    for did in def_id:
        # cell 1 is the number of missed shots when the player was guarding,
        # cell 2 is the total number of shots when the player was guarding
        def_dict[did] = [0.0,0.0]
    return off_dict, def_dict

def calculateMissedMadeShots(df):
    off_dict, def_dict = createDicts(df['player_id'].unique(),df['CLOSEST_DEFENDER_PLAYER_ID'].unique())
    for i, row in df.iterrows():
        #print(df['SHOT_RESULT'][i])
        if(df['SHOT_RESULT'][i] == 1):
            off_dict[row['player_id']][0] += 1
        else:
            def_dict[row['CLOSEST_DEFENDER_PLAYER_ID']][0] += 1
        off_dict[df['player_id'][i]][1] += 1
        def_dict[df['CLOSEST_DEFENDER_PLAYER_ID'][i]][1] += 1
    return off_dict, def_dict

def missedToNum(df):
    df['SHOT_RESULT_FAC'] = -1
    count = 0
    for i, row in df.iterrows():
        count += 1
        if(count%1000 == 0):
            print(count)
        if(row['SHOT_RESULT'] == '1'):
            df['SHOT_RESULT_FAC'][i] = 1
    return df
def factorizeData(df):
    df1=df
    off_dict, def_dict = calculateMissedMadeShots(df1)
    df1['made_avg'] = 0.0
    df1['missed_avg'] = 0.0
    count = 0
    for i, row in df1.iterrows():
        count += 1
        if(count %1000 == 0):
            print(count)
        off_made, off_total = off_dict[row['player_id']]
        def_missed, def_total = def_dict[row['CLOSEST_DEFENDER_PLAYER_ID']]
        #print(off_made/off_total)
        df1['made_avg'][i] = off_made/off_total
        df1['missed_avg'][i] = def_missed/def_total
    return df1

def calculateDefPlayerMissedAverage(df):
    defenders_id = df['CLOSEST_DEFENDER_PLAYER_ID'].unique()
    for i in defenders_id:
        df[df['CLOSEST_DEFENDER_PLAYER_ID']==i]

def createdf():
    with open('shot_logs.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        train ,test = [], []
        for row in reader:
            if(np.nan in row.values()):
                continue
            row['LOCATION'] = 1 if row['LOCATION']=='H' else -1
            row['W'] = 1 if row['W']=='W' else -1
            row['GAME_CLOCK'] = int(row['GAME_CLOCK'].split(':')[0])*60+int(row['GAME_CLOCK'].split(':')[1])
            row['SHOT_RESULT'] = 1 if row['SHOT_RESULT'] == 'made' else 0
            if(random.random() <= 0.8):
                train.append(row)
            else:
                test.append(row)
        df_train = pd.DataFrame(train)
        df_test = pd.DataFrame(test)
        df_test.to_csv("test_befor_factor.csv")
        df_train.to_csv("train_befor_factor.csv")
        return df_train, df_test

warnings.filterwarnings('ignore')
# train, test = createdf()
# factorized_train = factorizeData(train)
# factorized_test = factorizeData(test)
# factorized_train.to_csv('shot_log_train_factorized.csv')
# factorized_test.to_csv('shot_log_test_factorized.csv')

df_test = pd.read_csv('test_befor_factor.csv')
factorized_test = factorizeData(df_test)
factorized_test.to_csv('shot_log_test_factorized.csv')


"""
indexes = random.sample(range(1,len(train)),5000)
sample_train = []
sample_test = []
for i in range(4000):
    sample_train.append(train[indexes[i]])
for i in range(4000,5000):
    sample_test.append(train[indexes[i]])
print(sample_train[:5])

engine = create_engine('sqlite://', echo=False)
df.to_sql('train', con=engine)
"
print(engine.execute("SELECT *, AVG(SHOT_RESULT) as DEF_PLAYER_AVG FROM \
                    (SELECT *, AVG(SHOT_RESULT) as OFF_PLAYER_AVG FROM train GROUP BY player_id) \
                    GROUP BY CLOSEST_DEFENDER_PLAYER_ID ").fetchall()[0])


print(len(engine.execute("SELECT * FROM train").fetchall()))
"""