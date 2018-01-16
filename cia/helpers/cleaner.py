import numpy as np
import pandas as pd

def clean_data_using_labels(df, labels_original, interval, label_reducer):
    # Remove the rows where labels are missing
    # Labels are determined based on label reducer parameter
    missing_label_indexes = labels_original[np.isnan(labels_original['label'])].index
    remove_indexes = list(set(
        filter(
            lambda x: x >= 0, list(missing_label_indexes - interval) + list(missing_label_indexes)
        )
    ))
    label_indexes = labels_original.drop(pd.Int64Index(remove_indexes)).index
    labels_cleaned = labels_original.loc[label_indexes]

    labels_next_interval = reset_index(labels_original.loc[label_indexes + interval])
    labels_this_interval = reset_index(labels_cleaned)

    labels = label_reducer(labels_next_interval, labels_this_interval)
    df_cleaned = reset_index(df.loc[label_indexes])

    # Drop columns where 75% or more rows have NaN
    row_count = df_cleaned.shape[0]
    drop_column_threshold = (row_count * 0.25)

    column_indexes_to_drop = []

    nan_matrix = []

    for col_name, r in df_cleaned.T.iterrows():
        nan = r[np.isnan(r)]
        nan_count = len(nan)
        nan_matrix.append(nan.index)

        if nan_count >= drop_column_threshold:
            column_indexes_to_drop.append(col_name)

    # Fill in NaN with 0
    for column_index, row_indexes in enumerate(nan_matrix):
        df_cleaned.set_value(row_indexes, column_index, 0, takeable=True)

    # Drop all columns that have NaN values greater than threshold
    return df_cleaned.drop(column_indexes_to_drop, axis=1), labels

def reset_index(df):
    return df.reset_index().drop(['index'], axis=1)
