import pandas as pd
import numpy as np
import re

import requests
import sqlite3 

from time import sleep 
import tqdm 

from typing import Dict, List
import pickle

import redata
from redata.processing import *

NUM_PAGES = 10
SLEEP_TIME = 2

SAVE_INTERIM_TOP_PAGES_DOWNLOADS = True
SAVE_INTERIM_INDIVIDUAL_RE_DOWNLOADS = True
DATA_DIR = f'./data'

def download_and_extract_relevant_data(data_dir = './data', num_pages = 10, sleep_time = 2, save_interim_top_pages_downloads = True, save_interim_individual_re_downloads = True):
    """
    This function downloads real estate data from the website https://www.sreality.cz/. It takes four optional parameters:

    Args:
        data_dir: the directory where the data should be saved. The default value is './data'.
        num_pages: the number of pages of data to download. The default value is 10.
        sleep_time: the amount of time to sleep between requests. The default value is 2.
        save_interim_top_pages_downloads: a flag indicating whether or not to save intermediate data as the top pages are being downloaded. The default value is True.
        save_interim_individual_re_downloads: a flag indicating whether or not to save intermediate data as individual real estate listings are being downloaded. The default value is True.
    Returns:
        pd.DataFrame 

    Note: 
        Function is quite ugly and messy. Try refactoring it on your own :).
    """

    raw_collector = {}
    
    p = 0 
    for p in tqdm.tqdm(range(NUM_PAGES)):
        base_url = f'https://www.sreality.cz/api/cs/v2/estates?category_main_cb=1&category_type_cb=1&locality_region_id=10&per_page60&page={p}'
        print(base_url)

        r = requests.get(base_url)
        
        if 'message' in r.json():
            if (r.json()['message'] == 'Not found.'):
                quit()

        raw_collector[p] = r.json()

        if SAVE_INTERIM_TOP_PAGES_DOWNLOADS: 
            with open(f'{DATA_DIR}/raw/top_page_raw{p}.pickle', 'wb') as handle:
                pickle.dump(raw_collector[p], handle, protocol=pickle.HIGHEST_PROTOCOL)
        sleep(SLEEP_TIME)
    
    # save all data after downloading
    with open(f'{DATA_DIR}/raw/all_top_pages_raw.pickle', 'wb') as handle:
        pickle.dump(raw_collector[p], handle, protocol=pickle.HIGHEST_PROTOCOL)


    raw_attributes_to_collect = ['locality', 'price', 'name', 'gps','hash_id','labelsAll','exclusively_at_rk']

    estates_individual_raw = {}
    estates_individual = {}
    for page, r in raw_collector.items():
        for estate in r['_embedded']['estates']: 

            estate_raw = {k:v for k,v in estate.items() if k in raw_attributes_to_collect}
            estate_id = estate['hash_id']

            estate_relevant = pd.Series()


            estate_relevant['price'] = int(estate['price'])
            estate_relevant['area'] = get_area_from_name(estate['name'])

            lat, lon = get_gps_lat_lon(estate)
            estate_relevant.loc['gps_lat'] = lat
            estate_relevant.loc['gps_lon'] = lon
            estate_relevant['locality'] = estate['locality']

            estate_relevant['flat_type'] = get_flat_type_from_name(estate['name'])
            estate_relevant['hash_id'] = estate_id

            estates_individual[estate_id] = estate_relevant

            if SAVE_INTERIM_INDIVIDUAL_RE_DOWNLOADS:

                with open(f'{DATA_DIR}/individual/re_{estate_id}.pickle', 'wb') as handle:
                    pickle.dump(raw_collector[p], handle, protocol=pickle.HIGHEST_PROTOCOL)

    df = pd.concat(estates_individual).unstack()
    df['area'] = df['area'].astype(float)
    df.to_parquet(f'{DATA_DIR}/individual/re_all.parquet')

    return df