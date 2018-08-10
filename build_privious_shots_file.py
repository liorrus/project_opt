import numpy as np
import scipy as sc
import pandas as pd
import random
import sqlite3
import warnings
import csv
from sqlalchemy import create_engine

def sort_by_id_time(df):
    #df.sort_values(['a', 'b'], ascending=[True, False])
    df1=df.sort_values(by=['player_id','GAME_ID','PERIOD','GAME_CLOCK'], ascending=[True,True,True,False])
    return df1


def addFormerShotsDataFrame2(formerShots, df):
    dflist = list(df)
    for i in range(0, formerShots):
        j = i + 1
        for attribute in dflist:
            # print(attribute)
            st = attribute
            st = str(st)
            st += repr(j)  # add j to the end of string
            df[st] = df[attribute][1]
    for k in range(formerShots + 1, len(df)):
        for attribute in dflist:
            for i in range(0, formerShots):
                # print("kaki")
                j = i + 1
                st = attribute
                st = str(st)
                st += repr(j)
                df.loc[k, st] = df.loc[k-j,attribute]
                # print("value after change:", dfi[1][st].iloc[[k]])
    print(df[['SHOT_NUMBER', 'SHOT_NUMBER1', 'SHOT_NUMBER2']])
    df.to_csv('dfi1.csv')


df = pd.read_csv('shot_log_train_factorized1.csv')
#df1 = sort_by_id_time(df[:1000])
#addFormerShotsDataFrame(2,df)
addFormerShotsDataFrame2(2,df[:1000])
