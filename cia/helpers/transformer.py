import datetime
import numpy as np
import os
import pandas as pd

from helpers import common as cm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler

COLUMNS = [
    'created_at',
    'currency',
    'exchange',
    'price',
    'price_close',
    'price_high',
    'price_low',
    'price_open',
    'quantity',
    'timestamp',
    'timestamp_close',
    'timestamp_open',
    'uuid',
]

COLUMNS_AS_FEATURE = [
    'volume',
    'price_open',
    'price_close',
    'price_high',
    'price_low',
]

def add_time_component_columns(df):
    df['date'], df['hour'], df['minute'] = zip(
        *df['timestamp_close'].apply(extract_time_components)
    )
    return df

def extract_time_components(x):
    date = datetime.datetime.fromtimestamp(x)
    return int(date.strftime('%Y%m%d')), int(date.strftime('%H')), int(date.strftime('%M'))

def group_by(df, columns):
    group = df.groupby(columns, axis=0)
    return [(key, group.get_group(key)) for key in group.groups.keys()]

def group_by_date(df):
    return sorted(group_by(df, ['date']), key=lambda x: x[0])

def group_by_currency(df):
    return sorted(group_by(df, ['currency']), key=lambda x: x[0])

def group_by_hour(df):
    return sorted(group_by(df, ['hour']), key=lambda x: x[0])

def group_by_minute(df):
    return sorted(group_by(df, ['minute']), key=lambda x: x[0])

def transform_data_with_new_columns(df):
    group_sorted = df.sort_values(
        ['timestamp_close'], ascending=[1]
    ).drop_duplicates(
        'timestamp_close', keep='last'
    )
    volume = sum(group_sorted['quantity'])
    price_open = group_sorted.iloc[0]['price_open']
    closing_prices = group_sorted['price_close']
    price_close = closing_prices.iloc[len(group_sorted) - 1]
    price_high = max(closing_prices)
    price_low = min(closing_prices)
    return volume, price_open, price_close, price_high, price_low

def transform_all(df):
    new_columns = [
        'date',
        'hour',
        'minute',
        'currency',
        'volume',
        'price_open',
        'price_close',
        'price_high',
        'price_low',
    ]
    d = {}
    count = 0

    for currency, g_by_c in group_by_currency(df):
        arr = []
        for date, g_by_d in group_by_date(g_by_c):
            for hour, g_by_h in group_by_hour(g_by_d):
                for minute, g_by_m in group_by_minute(g_by_h):
                    values = transform_data_with_new_columns(g_by_m)
                    arr.append((date, hour, minute, currency) + values)
                    count += 1
                    if count % 100000 == 0:
                        print(count)
        d[currency] = pd.DataFrame(data=arr, columns=new_columns)

    return d

def save_currencies(df1, exchange):
    for currency, df2 in df1.items():
        df2.to_csv('data/currencies/{}/{}.csv'.format(exchange, currency), index=False)

def load_transformed_data_from_currencies(exchange):
    data = {}
    currencies = []
    for name in os.listdir('data/currencies/{}'.format(exchange)):
        if not name.startswith('.') :
            currencies.append(name.split('.csv')[0])

    for currency in currencies:
        data[currency] = cm.load_data(
            'data/currencies/{}/{}.csv'.format(exchange, currency)
        )

    return data

class DateFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attribute_names].values

def build_pipeline(numerical_attributes):
    scalar_pipeline = Pipeline([
        ('selector', DateFrameSelector(numerical_attributes)),
        ('standard_scalar', StandardScaler()),
    ])
    return FeatureUnion(transformer_list=[
        ('scalar_pipeline', scalar_pipeline),
    ])

def scale_data(df):
    pipeline = build_pipeline(COLUMNS_AS_FEATURE)
    df_scaled = pd.DataFrame(data=pipeline.fit_transform(df), columns=COLUMNS_AS_FEATURE)
    df_scaled[['date', 'hour_of_day', 'minute_of_hour']] = df[['date', 'hour', 'minute']]
    return df_scaled

def scaled_and_original_data(df_t):
    scaled_data = {}
    original_data = {}
    count = 0

    for currency, df in df_t.items():
        scaled_data[currency] = {}
        original_data[currency] = {}

        shape = df.shape
        rows = shape[0]

        df_scaled = scale_data(df)
        pairs = [
            (df_scaled, scaled_data, ('date', 'hour_of_day', 'minute_of_hour')),
            (df, original_data, ('date', 'hour', 'minute')),
        ]

        for i in range(0, rows):
            for d_frame, d, cols in pairs:
                row = d_frame.iloc[i]

                date = row[cols[0]]
                hour_of_day = row[cols[1]]
                minute_of_hour = row[cols[2]]

                if not d[currency].get(date, False):
                    d[currency][date] = {}

                if not d[currency][date].get(hour_of_day, False):
                    d[currency][date][hour_of_day] = {}

                d[currency][date][hour_of_day][minute_of_hour] = row[COLUMNS_AS_FEATURE]

            count += 1
            if count % 100000 == 0:
                print(count)

    return scaled_data, original_data

def scaled_and_original_vectors(scaled_data, original_data):
    vectors = []
    vectors_original = []

    currencies = sorted(scaled_data.keys())
    all_dates = sorted(list(set(
        cm.flatten([scaled_data[curr].keys() for curr in currencies])
    )))
    all_hours = cm.all_hours()
    all_minutes = cm.all_minutes()

    hours = np.ndarray(shape=(24, 1), dtype=np.float64)
    for i in range(0, 24):
        hours[i] = np.array([i], dtype=np.float64)
    hours_scaled = StandardScaler().fit_transform(hours)

    minutes = np.ndarray(shape=(60, 1), dtype=np.float64)
    for i in range(0, 60):
        minutes[i] = np.array([i], dtype=np.float64)
    minutes_scaled = StandardScaler().fit_transform(minutes)

    for date in all_dates:
        for hour in all_hours:
            for minute in all_minutes:
                vector = [
                    date,
                    hour,
                    minute,
                    hours_scaled[int(hour)][0],
                    minutes_scaled[int(minute)][0],
                ]
                vector_original = list(vector)

                for currency in currencies:
                    d1 = scaled_data[currency]
                    try:
                        vector += list(d1[date][hour][minute])
                    except KeyError:
                        vector += [np.nan for i in range(0, len(COLUMNS_AS_FEATURE))]

                    d2 = original_data[currency]
                    try:
                        vector_original += list(d2[date][hour][minute])
                    except KeyError:
                        vector_original += [np.nan for i in range(0, len(COLUMNS_AS_FEATURE))]

                vectors.append(vector)
                vectors_original.append(vector_original)

    return vectors, vectors_original

def save_transformed_and_original(vectors, vectors_original, exchange, currencies):
    shared_initial_columns = [
        'date',
        'hour',
        'minute',
        'hour_scaled',
        'minute_scaled',
    ]
    currency_feature_columns = cm.flatten(
        [['{}_{}'.format(col_name, curr) for curr in currencies for col_name in COLUMNS_AS_FEATURE]]
    )
    df_shared_columns = shared_initial_columns + currency_feature_columns

    df_shared_new = pd.DataFrame(vectors, columns=df_shared_columns)
    df_shared_new.to_csv('data/shared/{}/transformed.csv'.format(exchange), index=False)

    df_shared_original_new = pd.DataFrame(vectors_original, columns=df_shared_columns)
    df_shared_original_new.to_csv('data/shared/{}/original.csv'.format(exchange), index=False)

    return df_shared_new, df_shared_original_new

def transform_and_scale(exchange):
    df_raw = cm.load_data('datasets/{}Chart'.format(exchange))
    df_raw.columns = COLUMNS

    df_raw_with_time = add_time_component_columns(df_raw)
    df_transformed_initial_by_currency = transform_all(df_raw_with_time)
    save_currencies(df_transformed_initial_by_currency, exchange)

    scaled_data, original_data = scaled_and_original_data(
        load_transformed_data_from_currencies('Bittrex'),
    )
    vectors, vectors_original = scaled_and_original_vectors(scaled_data, original_data)

    currencies = cm.all_currencies('data/currencies/{}'.format(exchange))
    save_transformed_and_original(vectors, vectors_original, exchange, currencies)
