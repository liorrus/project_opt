import csv
with open('dfi2.csv', 'r',newline='') as csvfile:
    rows = []
    reader = csv.reader(csvfile,delimiter=',')
    writer = csv.writer(open('dfi2_no_nan.csv', 'w',newline=''),delimiter=',')
    for row in reader:
        if("" in row):
            continue
        writer.writerow(row)