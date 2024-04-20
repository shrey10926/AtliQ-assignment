from pickle import load
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, make_response, abort
import logging
from flask_cors import CORS

log_format = "%(asctime)s::%(name)s::"\
             "%(filename)s::%(funcName)s::%(lineno)d::%(message)s"
logging.basicConfig(filename='log_file.log', filemode='w', level='DEBUG', format=log_format)
logger = logging.getLogger()


app = Flask(__name__)
CORS(app)


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def custom_abort(message,code):

    if code == 200:
        json = jsonify(message)
        response = make_response(json,200)
        return abort(response)

    else:
        json = jsonify(errorMessage = message)
        response = make_response(json,400)
        return abort(response)


def separate_df(data):  
  num = data.select_dtypes(include = 'number')
  cat = data.select_dtypes(include = 'object')
  cat = cat.reset_index(drop = True)
  return num, cat


@app.route('/predict', methods = ['POST'])
def flask_api():

  if request.method == 'OPTIONS':
      return _build_cors_preflight_response()


  try:
      logger.debug('Fetching data from request...')
      data = request.get_json()
      logger.debug('Data Fetched!')

  except Exception as e:
    logger.debug(e)
    return custom_abort(str(e), 400)


  for k, v in data.items():


    try:
      # load the model
      model = load(open('xgb_model.pkl', 'rb'))
      logger.debug('Model loaded!')
      # load the scaler
      scaler = load(open('scaler.pkl', 'rb'))
      logger.debug('Scaler loaded')
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      test_df = pd.DataFrame([data.get('entries')], columns =['area_code', 'transit_server_type',	'log_report_type',	'volume',	'broadband_type',	'outage_type'])
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      num, cat = separate_df(test_df)
      test_df = pd.concat([num, cat], axis = 1)    
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      nutest_dfm = pd.DataFrame(scaler.transform(test_df), columns = test_df.columns)
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)

    try:
      final_pred = model.predict(nutest_dfm)
    except Exception as e:
      logger.debug(e)
      return custom_abort(str(e), 400)


    if final_pred == 0:
      return f'No Outage'
    elif final_pred == 1:
      return 'SHort Outage'
    elif final_pred == 2:
      return 'Long Outage'
    else:
      return 'Unknown Outage'

  return custom_abort('Done!', 200)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = False, port = 6996)