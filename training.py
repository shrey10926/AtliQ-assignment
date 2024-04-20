import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
# import category_encoders as ce
from sklearn.metrics import f1_score
from pickle import dump
import xgboost as xgb
from matplotlib import pyplot
# from xgboost import plot_importance
# from xgboost import XGBClassifier



df = pd.read_csv('data/train_merged.csv')

#Taking deep copy
dff = df.copy(deep = True)

#reshuffle the data
df1 = dff.sample(frac = 1, random_state = 69).reset_index(drop = True)

#No duplicates
df1[df1.duplicated(keep = 'last')]


#remove the constant sub-string. No need for encoders.
df1['area_code'] = df1['area_code'].str.replace('area_', '').astype(int)
df1['transit_server_type'] = df1['transit_server_type'].str.replace('transit_server_type_', '').astype(int)
df1['log_report_type'] = df1['log_report_type'].str.replace('log_report_type_', '').astype(int)
df1['broadband_type'] = df1['broadband_type'].str.replace('broadband_type_', '').astype(int)
df1['outage_type'] = df1['outage_type'].str.replace('outage_type_', '').astype(int)


#remove id as no relevence
df1.drop(['id'], axis = 1, inplace = True)


#split data
def target_encode_split(dataframe):
  
  x = dataframe.drop(['outage_duration'], axis = 1)
  y = dataframe['outage_duration']
  trainx, testx, trainy, testy = train_test_split(x, y, test_size = 0.2, random_state = 69, stratify = y)

  df_list = [trainx, testx, trainy, testy]

  for i in df_list:
    i.reset_index(drop = True, inplace = True)
  
  return trainx, testx, trainy, testy

train_x, test_x, train_y, test_y = target_encode_split(df1)



#dropping log_report_type reduces the accuracy. Hence not performing feature selection
########################################################################################################
# #Feature Selection
# model = XGBClassifier()
# model.fit(train_x, train_y)

# # feature importance
# print(model.feature_importances_)
# plot_importance(model)
# pyplot.show()

# train_x.drop(['log_report_type'], axis = 1, inplace = True)
# test_x.drop(['log_report_type'], axis = 1, inplace = True)
########################################################################################################


#separate columns based on dtypes
def separate_dtypes(trx, tex):
  train_num = trx.select_dtypes(include = 'number')
  train_cat = trx.select_dtypes(include = 'object')
  train_cat = train_cat.reset_index(drop = True)

  test_num = tex.select_dtypes(include = 'number')
  test_cat = tex.select_dtypes(include = 'object')
  test_cat = test_cat.reset_index(drop = True)

  return train_num, train_cat, test_num, test_cat

train_num, train_cat, test_num, test_cat = separate_dtypes(train_x, test_x)


#scale data
def scale(trx3, tex3):
  scaler = StandardScaler()
  scaler.fit(trx3)
  trx3 = pd.DataFrame(scaler.transform(trx3), columns = trx3.columns)
  tex3 = pd.DataFrame(scaler.transform(tex3), columns = tex3.columns)

  return trx3, tex3, scaler

train_x3, test_x3, scaler = scale(train_num, test_num)



#define models
dtree = DecisionTreeClassifier(random_state = 69)
rforest = RandomForestClassifier(random_state = 69)
xgbc = xgb.XGBClassifier()

#define parameter space
dtree_param = [{'max_depth' : [3, 4, 5], 'max_features' : ['sqrt', 'log2']}]
rforest_param = [{'n_estimators' : [5, 10, 20], 'max_depth' : [3, 4, 5]}]
XGBC_params = [{'eta': [0.1, 0.2, 0.3], 'n_estimators' : [10, 50, 100], 'max_depth': [3, 6, 9]}]


#nested cv
inner_cv = KFold(n_splits = 3)
gridcvs = {}

# estimate performance of hyperparameter tuning and model algorithm pipeline
for params, model, name in zip((dtree_param, rforest_param, XGBC_params), (dtree, rforest, xgbc), ('DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier')):

    # perform hyperparameter tuning
    gcv = GridSearchCV(estimator = model, param_grid = params, cv = inner_cv, 
                       scoring = 'f1_micro',
                       refit = True)
    
    gridcvs[name] = gcv


outer_cv = KFold(n_splits = 5)

# outer loop cv
for name, gs_model in sorted(gridcvs.items()):
      nested_score = cross_val_score(gs_model, train_x3, train_y, 
                                     cv = outer_cv, n_jobs = -1, 
                                     scoring = 'f1_micro')
      print(name, nested_score.mean(), nested_score.std())


# select HP for the best model based on regular k-fold on whole training set    
final_cv = KFold(n_splits = 5)

gcv_final_HP = GridSearchCV(estimator = xgbc,
                            param_grid = XGBC_params,
                            cv = final_cv, scoring = 'f1_micro'
                            )
    
gcv_final_HP.fit(train_x3, train_y)



# check the score of normal gridsearchcv and compare that to the best model selected in nestedcv
# if the score are very different then bias has been introduced. In this case, the scores are exactly similar.
gcv_final_HP.best_score_

# get the best params from the gcv_final_HP
gcv_final_HP.best_params_


final_model = xgb.XGBClassifier(eta = 0.3, max_depth = 9, n_estimators = 100)
# fit the model to whole "training" dataset
final_model.fit(train_x3, train_y)
pred = final_model.predict(test_x3)

c_matrix = confusion_matrix(test_y, pred)
#83.6
f1score = f1_score(test_y, pred, average='micro')

# save the model
dump(final_model, open('xgb_model.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))