import numpy as np
import scipy as sc
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv('shot_log_train_factorized_no_nan.csv')

sample = df.sample(n=10000)
columns = sample.columns.difference(['player_name', 'player_id', 'CLOSEST_DEFENDER',
                    'CLOSEST_DEFENDER_PLAYER_ID', 'GAME_ID','MATCHUP','SHOT_RESULT',
                                   'made_avg', 'missed_avg'])
train_sample = df.sample(1000)
test_sample = df.sample(1000)
train_results = np.array(train_sample['SHOT_RESULT'])
train_features = train_sample[columns]
# print(train_features[0])
test_results = np.array(test_sample['SHOT_RESULT'])
test_features = test_sample[columns]
# print(test_features[0])

clf = SVC(kernel="linear")
print("started fit")
clf.fit(train_features, train_results)
predictions = clf.predict(test_features)
true_p = 0
false_p = 0
true_n = 0
false_n = 0
for i in range(len(predictions)):
    if(predictions[i] == 1 and test_results[i] == 1):
        true_p += 1
    elif(predictions[i] == 0 and test_results[i] == 0):
        true_n += 1
    elif(predictions[i] == 1 and test_results[i] == 0):
        false_p += 1
    elif (predictions[i] == 0 and test_results[i] == 1):
        false_n += 1
print((true_p+true_n)/len(predictions), (false_p+false_n)/len(predictions))
print("tp", true_p/len(predictions))
print("tn", true_n/len(predictions))
print("fp", false_p/len(predictions))
print("fn", false_n/len(predictions))
