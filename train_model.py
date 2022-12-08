import pandas as pd
import lightgbm
import logging

import redata
from redata.processing import *

from sklearn.preprocessing import OrdinalEncoder
import joblib


if __name__ == '__main__':

    ### CONFIGURATION VARIABLES - could be loaded from configuration file, e.g. yaml/json/etc.
    FEATURES = ['area','flat_type','prague_part']
    LABEL = ['price']
    CATEG_FEATURES = ['flat_type','prague_part']

    # default hyperparameters
    MODEL = lightgbm.LGBMRegressor()

    
    LOGS_DIR = 'logs'
    MODELS_DIR = 'models'

    logging.basicConfig(filename=f'{LOGS_DIR}/model_trainning.log',
        filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    logging.info("Loading previously collected data.")
    data = pd.read_parquet('data/individual/re_all.parquet')
    data_shape = data.shape
    logging.info(f"Loaded data with shape {data_shape}.")


    data['prague_part'] = data['locality'].apply(lambda x: get_general_prague_part(x))
    data = data.loc[data['price'] > 1000,:]

    enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value = -1)
    enc.fit(data[['prague_part','flat_type']])
    data[['prague_part','flat_type']] = enc.transform(data[['prague_part','flat_type']])
    joblib.dump(enc,f'{MODELS_DIR}/enc.joblib')

    model = lightgbm.LGBMRegressor()
    model.fit(data[FEATURES], data[LABEL])
    joblib.dump(model,f'{MODELS_DIR}/model.joblib')
