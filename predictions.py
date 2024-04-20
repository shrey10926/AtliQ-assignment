import pandas as pd
from pickle import load

# Assuming 'model' is your trained model and 'merged_test_data' is your merged test data DataFrame

test_df = pd.read_csv('data/test_merged.csv')

#remove duplicates
test_df[test_df.duplicated(keep = 'last')]


#remove the constant sub-string. No need for encoders.
test_df['area_code'] = test_df['area_code'].str.replace('area_', '').astype(int)
test_df['transit_server_type'] = test_df['transit_server_type'].str.replace('transit_server_type_', '').astype(int)
test_df['log_report_type'] = test_df['log_report_type'].str.replace('log_report_type_', '').astype(int)
test_df['broadband_type'] = test_df['broadband_type'].str.replace('broadband_type_', '').astype(int)
test_df['outage_type'] = test_df['outage_type'].str.replace('outage_type_', '').astype(int)

test_df.drop(['id'], axis = 1, inplace=True)


# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns = test_df.columns)


# load the model
model = load(open('xgb_model.pkl', 'rb'))
final_pred = model.predict(test_df_scaled)
test_df['predictions'] = final_pred
test_df.to_csv('data/test_predictions.csv', index=False)