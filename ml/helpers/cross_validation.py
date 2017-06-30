from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

def cross_validate(opts={}):
    """
    data: examples to train with
    labels: labels of examples
    model: ML model
    n_splits (optional): # of cross validation
    """
    
    data = opts['data']
    labels = opts['labels']
    scores = []
    
    skfolds = StratifiedKFold(
        n_splits=opts.get('n_splits', 3),
        random_state=42
    )
    
    for train_idx, test_idx in skfolds.split(data, labels):
        model_clone = clone(opts['model'])
        
        X_train = data[train_idx]
        y_train = labels[train_idx]
        X_test = data[test_idx]
        y_test = labels[test_idx]
        
        model_clone.fit(X_train, y_train)
        predictions = model_clone.predict(X_test)
        correct = sum(predictions == y_test)
        print(correct / len(predictions))