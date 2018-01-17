import math
import numpy as np
import pandas as pd

from helpers import common as cm
from helpers import cleaner as cl

from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit

def training_and_test_sets(df_input, labels_input, columns_to_stratify):
    df = cl.reset_index(df_input)
    labels_final = cl.reset_index(labels_input)

    # Split data
    stratified_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    # for train_index, test_index in
    if columns_to_stratify:
        gen = stratified_split.split(df, df[columns_to_stratify])
    else:
        gen = stratified_split.split(df, df[['hour_scaled', 'minute_scaled']])

    for training_indices, test_indices in gen:
        training_set = df.loc[training_indices]
        if columns_to_stratify:
            training_set = training_set.drop(columns_to_stratify, axis=1)
        training_set_labels = labels_final.loc[training_indices]

        test_set = df.loc[test_indices]
        if columns_to_stratify:
            test_set = test_set.drop(columns_to_stratify, axis=1)
        test_set_labels = labels_final.loc[test_indices]

    return training_set, training_set_labels, test_set, test_set_labels

from sklearn.model_selection import GridSearchCV

def train_using_best_features(opts={}):
    coef_extractor = opts['coef_extractor']
    features_to_use = opts['features_to_use']

    gs = opts.get('grid_search', None)
    model = opts.get('model', None)

    training_set, training_set_labels, test_set, test_set_labels = training_and_test_sets(
        opts['data'],
        opts['labels'],
        opts['columns_to_stratify'],
    )

    training_set_to_use = training_set[features_to_use]
    test_set_to_use = test_set[features_to_use]

    if gs:
        gs.fit(training_set_to_use, cm.labels_to_array(training_set_labels))
        model = gs.best_estimator_
        score = model.score(test_set_to_use, cm.labels_to_array(test_set_labels))
    else:
        scorer = opts['scorer']
        model.fit(training_set_to_use, cm.labels_to_array(training_set_labels))
        score = scorer(cm.labels_to_array(test_set_labels), model.predict(test_set_to_use))

    feature_importances = sorted(
        zip(coef_extractor(model), features_to_use),
        key=lambda x: np.absolute(x[0]),
        reverse=True,
    )

    return score, model, feature_importances, [f for s, f in feature_importances]

def train(training_opts):
    features_to_use = training_opts['features_to_use']
    max_index = len(features_to_use) - 1

    power = 2
    max_iter = int(math.ceil(max_index**(1.0 / power)))

    n_features_to_try = training_opts.get(
        'n_features_to_try',
        sorted(
            list(set([int(math.floor((i**power))) for i in range(1, max_iter)] + [max_index])),
            reverse=True,
        ),
    )

    scores_and_features = []

    for i in n_features_to_try:
        training_opts['features_to_use'] = features_to_use[:i]
        score,  model, feature_importances, features_to_use = train_using_best_features(
            training_opts,
        )
        print('{}: {}'.format(len(feature_importances), score))
        scores_and_features.append((score, model, feature_importances, features_to_use))

    best_scores = sorted(scores_and_features, key=lambda x: x[0], reverse=True)

    for tup in best_scores[0:3]:
        print('\n')
        print('Top 3:')
        print(tup[0], len(tup[2]))

    return best_scores[0]

def save_model(model, model_name):
    joblib.dump(model, 'models/{}.pkl'.format(model_name))

def load_model(model_name):
    return joblib.load('models/{}.pkl'.format(model_name))

def save_features(model_name, features):
    pd.DataFrame(features, columns=['weight', 'feature']).to_csv(
        'models/{}_features.csv'.format(model_name),
        index=False,
    )
