import os
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from itertools import product
from readers import txt_linesreader

# Labeling:
# 0 - same author
# 1 - different author

# Default classifier
default_clf = DecisionTreeClassifier()


def compute_metrics(author1_data: list, author2_data: list, clf=default_clf):
    """
    Train a classifier and compute evaluation metrics.

    :param author1_data: Feature vectors for same author (label 0)
    :param author2_data: Feature vectors for different author (label 1)
    :param clf: Classifier object (default: DecisionTreeClassifier)
    :return: Tuple of (accuracy, precision, recall, confusion_matrix)
    """
    data_x = author1_data + author2_data
    data_y = [0] * len(author1_data) + [1] * len(author2_data)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        confusion_matrix(y_test, y_pred)
    )


def predict(author1_data: list, author2_data: list, x_test: list, clf=default_clf):
    """
    Train classifier on training data and predict labels for new samples.

    :param author1_data: Samples for label 0
    :param author2_data: Samples for label 1
    :param x_test: Samples to predict
    :param clf: Classifier object
    :return: List of predicted labels
    """
    x_train = author1_data + author2_data
    y_train = [0] * len(author1_data) + [1] * len(author2_data)

    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def load_feature_vectors(folder_path: str):
    """
    Load and parse feature vectors from a directory structure.

    :param folder_path: Root directory containing 'auth1–auth1' and 'auth1–auth2' subfolders.
    :return: Tuple of (auth1_auth1_data, auth1_auth2_data)
    """
    aurtor1_author1_data = []
    author1_author2_data = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden/system files
            stats_raw = txt_linesreader(str(os.path.join(root, file)))[0]
            stats = eval(stats_raw)

            values = [list(sample.values()) for sample in stats.values()]
            if 'auth1–auth1' in root:
                aurtor1_author1_data.extend(values)
            elif 'auth1–auth2' in root:
                author1_author2_data.extend(values)

    return aurtor1_author1_data, author1_author2_data


def measure_model_performance(author1_author1_data, author1_author2_data, n_trials=100, clf=default_clf):
    """
    Measure model performance over multiple trials and compute average metrics.

    :param author1_author1_data: list of lists containing info from the same authors
    :param author1_author2_data: list of lists containing info from different authors
    :param author1_author1_data: Samples for label 0
    :param author1_author2_data: Samples for label 1
    :param n_trials: Number of times to repeat training/testing
    :param clf: Classifier object
    :return: Dictionary with mean accuracy, precision, and recall
    """
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for _ in range(n_trials):
        acc, prec, rec, _ = compute_metrics(author1_author1_data, author1_author2_data, clf)
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)

    return {
        'accuracy': statistics.mean(accuracy_scores),
        'precision': statistics.mean(precision_scores),
        'recall': statistics.mean(recall_scores)
    }


# n = [2, 3, 4]
# normalisation = [True, False]
# configurations = product(n, normalisation)
#
# for config in configurations:
#     path = '/Users/ivanguseff/PycharmProjects/LitSim/'
#
#     if config[1] is True:
#         path = os.path.join(path, 'results_normalised', f'N={config[0]}')
#     else:
#         path = os.path.join(path, 'results', f'N={config[0]}')
#
#     vectors_auth1_auth_1, vectors_auth1_auth_2 = (
#         load_feature_vectors(path))
#     print(config)
#     print(measure_model_performance(vectors_auth1_auth_1, vectors_auth1_auth_2))
