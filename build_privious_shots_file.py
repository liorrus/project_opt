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
 
def addFormerShotsDataFrame(formerShots,df):
    player_list=df.groupby('player_id')
    count = 0
    for dfi in player_list:
        count+=1
        print(dfi)
        if count == 3 :
            break
    # for i in xrange(formerShots):


df = pd.read_csv('shot_log_train_factorized.csv')
df1 = sort_by_id_time(df)
addFormerShotsDataFrame(1,df1)
print(list(df1))
#df2=df1[:1000]