import numpy as np

def split_train_test(data_frame, ratio_of_test_set):
    """
    Split the training data into a set used to train the model and a set for testing the model.
    """
    m_training_examples = len(data_frame)
    test_set_size = int(m_training_examples * ratio_of_test_set)

    shuffled_indices = np.random.permutation(m_training_examples)
    training_indices = shuffled_indices[test_set_size:]
    test_indices = shuffled_indices[:test_set_size]

    return data_frame.iloc[training_indices], data_frame.iloc[test_indices]

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
