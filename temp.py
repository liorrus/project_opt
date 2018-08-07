import csv
with open('shot_log_train_factorized.csv', 'r',newline='') as csvfile:
    rows = []
    # fieldnames = []
    # for head in headers:
    #     if head not in ['player_name', 'player_id', 'CLOSEST_DEFENDER',
    #                 'CLOSEST_DEFENDER_PLAYER_ID', 'GAME_ID','MATCHUP', '']:
    #         fieldnames.append(head)
    reader = csv.reader(csvfile,delimiter=',')
    writer = csv.writer(open('shot_log_train_factorized_no_nan.csv', 'w',newline=''),delimiter=',')
    for row in reader:
        if("" in row):
            continue
        writer.writerow(row)