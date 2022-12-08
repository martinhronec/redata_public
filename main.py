import pandas as pd
import joblib

import redata
from redata.downloading import download_and_extract_relevant_data
from redata.processing import get_general_prague_part

import logging


if __name__ == '__main__':

    LOGS_DIR = './logs'

    ### CONFIGURATION VALUES, could be in external json
    NUM_PAGES = 10
    SLEEP_TIME = 2

    SAVE_INTERIM_TOP_PAGES_DOWNLOADS = True
    SAVE_INTERIM_INDIVIDUAL_RE_DOWNLOADS = True
    DATA_DIR = f'./data'
    
    TOP_N = 10
    MODELS_DIR = 'models'
    FEATURES = ['area','flat_type','prague_part']

    data = download_and_extract_relevant_data(
        data_dir = DATA_DIR, num_pages = NUM_PAGES,
        sleep_time = 2, save_interim_top_pages_downloads = SAVE_INTERIM_TOP_PAGES_DOWNLOADS,
        save_interim_individual_re_downloads = SAVE_INTERIM_INDIVIDUAL_RE_DOWNLOADS
        )
    #data = pd.read_parquet(f'{DATA_DIR}/individual/re_all.parquet')

    logging.basicConfig(filename=f'{LOGS_DIR}/main.log',
        filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    logging.info("Loading previously trained model and categories encoder.")
    enc = joblib.load(f'{MODELS_DIR}/enc.joblib')
    model = joblib.load(f'{MODELS_DIR}/model.joblib')

    logging.info("Cleaning data and extracting relevant features.")
    data = data.loc[data['price'] > 1000,:]
    data['prague_part'] = data['locality'].apply(lambda x: get_general_prague_part(x)).astype('str')
    data['flat_type'] = data['flat_type'].astype(str)
    data[['prague_part','flat_type']] = enc.transform(data[['prague_part','flat_type']])

    logging.info('Predicting values.')
    data['prediction'] = model.predict(data[FEATURES])
    data['diff'] = data['prediction'] - data['price']

    logging.info('Show real estate with the highest')
    print(data.sort_values('diff', ascending=False).head(TOP_N))