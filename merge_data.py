import pandas as pd

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
broadband = pd.read_csv('data/broadband_data.csv')
report = pd.read_csv('data/report_data.csv')
server = pd.read_csv('data/server_data.csv')
outage = pd.read_csv('data/outage_data.csv')


train_merged = pd.merge(train, server, on='id', how='left')
train_merged = pd.merge(train_merged, report, on='id', how='left')
train_merged = pd.merge(train_merged, broadband, on='id', how='left')
train_merged = pd.merge(train_merged, outage, on='id', how='left')
train_merged.to_csv('data/train_merged.csv', index = False)


test_merged = pd.merge(test, server, on='id', how='left')
test_merged = pd.merge(test_merged, report, on='id', how='left')
test_merged = pd.merge(test_merged, broadband, on='id', how='left')
test_merged = pd.merge(test_merged, outage, on='id', how='left')
test_merged.to_csv('data/test_merged.csv', index = False)