import pandas as pd
import numpy as np
import re

import requests
import sqlite3 

from time import sleep 
import tqdm 

from typing import Dict, List

def get_gps_lat_lon(estate_raw: Dict):
    """Docstrings are cool!"""
    gps_ = estate_raw['gps']
    return gps_['lat'], gps_['lon']

def get_flat_type_from_name(name: str):
    # Name is always represented by string "Prodej bytu [type of flat] [Area] m^2"
    return name.split()[2]

def get_area_from_name_naive(name: str):
    return int(name.split()[3])

# more robust implementation using regex - https://docs.python.org/3/library/re.html
def get_area_from_name(name: str):
    """(\d*) matches any decimal digits (repeated after each other"""
    name_ = name.split()
    return int(''.join(re.findall('(\d*)', ''.join(name_[3:]))))

# slow(er) helper function (use itertools.chains for speedup)
def flatten_list(l:List):
    return [item for sublist in l for item in sublist]

def get_distinct_nearby_objects(estates_individual_raw: Dict):
    lists_of_nearby_objects = [e['labelsAll'][1] for e in estates_individual_raw.values()]
    return list(set(flatten_list(lists_of_nearby_objects)))

def get_distinct_estate_attributes(estates_individual_raw: Dict):
    lists_of_estates_attributes = [e['labelsAll'][0] for e in estates_individual_raw.values()]
    return list(set(flatten_list(lists_of_estates_attributes)))

def get_general_prague_part(prague_part_detailed: str):
    return prague_part_detailed.split('-')[0][:-1]