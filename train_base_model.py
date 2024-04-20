import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


df = pd.read_csv('data/train_merged.csv')

x = df.drop(['outage_duration'], axis = 1)
y = df['outage_duration']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 69, stratify = y)

for i in [train_x, test_x, train_y, test_y]:
    i.reset_index(inplace = True , drop = True)

train_num = train_x.select_dtypes(include = 'number')
train_cat = train_x.select_dtypes(include = 'object')

test_num = test_x.select_dtypes(include = 'number')
test_cat = test_x.select_dtypes(include = 'object')


ohe = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
ohe.fit(train_cat)
train_nom_features = pd.DataFrame(ohe.transform(train_cat), columns = ohe.get_feature_names_out())
test_nom_features = pd.DataFrame(ohe.transform(test_cat), columns = ohe.get_feature_names_out())


train_x1 = pd.concat([train_num, train_nom_features], axis = 1)
test_x1 = pd.concat([test_num, test_nom_features], axis = 1) 


scaler = StandardScaler()
scaler.fit(train_x1)
train_x1 = pd.DataFrame(scaler.transform(train_x1), columns = train_x1.columns)
test_x1 = pd.DataFrame(scaler.transform(test_x1), columns = test_x1.columns)


model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()
model4 = KNeighborsClassifier()
model5 = xgb.XGBClassifier()


model1.fit(train_x1, train_y)
pred1 = model1.predict(test_x1)

acc1 = accuracy_score(test_y, pred1)
prec = precision_score(test_y, pred1, average = 'micro')
recall = recall_score(test_y, pred1, average = 'micro')
f1 = f1_score(test_y, pred1, average = 'micro')



model2.fit(train_x1, train_y)
pred2 = model2.predict(test_x1)

acc2 = accuracy_score(test_y, pred2)
prec = precision_score(test_y, pred2, average = 'micro')
recall = recall_score(test_y, pred2, average = 'micro')
f1 = f1_score(test_y, pred2, average = 'micro')



model3.fit(train_x1, train_y)
pred3 = model3.predict(test_x1)

acc3 = accuracy_score(test_y, pred3)
prec = precision_score(test_y, pred3, average = 'micro')
recall = recall_score(test_y, pred3, average = 'micro')
f1 = f1_score(test_y, pred3, average = 'micro')



model4.fit(train_x1, train_y)
pred4 = model4.predict(test_x1)

acc4 = accuracy_score(test_y, pred4)
prec = precision_score(test_y, pred4, average = 'micro')
recall = recall_score(test_y, pred4, average = 'micro')
f1 = f1_score(test_y, pred4, average = 'micro')


model5.fit(train_x1, train_y)
pred6 = model5.predict(test_x1)

c_matrix6 = confusion_matrix(test_y, pred6)
acc6 = accuracy_score(test_y, pred6)
prec = precision_score(test_y, pred6, average = 'micro')
recall = recall_score(test_y, pred6, average = 'micro')
f1 = f1_score(test_y, pred6, average = 'micro')




#TOP3 MODELS ARE DTC, RF, XGB






