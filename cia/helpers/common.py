import os
import pandas as pd

def all_currencies(file_path='data/currencies/Bittrex'):
    df_transformed = {}
    currencies = []

    for name in os.listdir(file_path):
        if not name.startswith('.') :
            currencies.append(name.split('.csv')[0])

    return sorted(currencies)

def all_hours():
    return [i for i in range(0, 24)]

def all_minutes():
    return [i for i in range(0, 60)]

def flatten(l):
    return [item for sublist in l for item in sublist]

def labels_to_array(df):
    return list(df['label'])

def load_data(file_path):
    return pd.read_csv(file_path)

