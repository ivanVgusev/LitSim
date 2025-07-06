from sklearn.neighbors import KNeighborsClassifier  # К-ближайших соседей
from sklearn.tree import DecisionTreeClassifier  # дерево принятия решений
from sklearn.svm import SVC  # метод опорных векторов
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # оценка
import matplotlib.pyplot as plt
import numpy as np

from readers import txt_linesreader
import os
import statistics

"""
0 - same author
1 - different author
"""

clf = DecisionTreeClassifier()


def training(author1_author1, author1_author2):
    data_x = author1_author1 + author1_author2
    data_y = [0 for _ in range(len(auth1_auth1))] + [1 for _ in range(len(auth1_auth2))]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)

    accuracy_s = accuracy_score(y_test, y_prediction)
    precision_s = precision_score(y_test, y_prediction)
    recall_s = recall_score(y_test, y_prediction)
    conf_matrix_s = confusion_matrix(y_test, y_prediction)

    return accuracy_s, precision_s, recall_s, conf_matrix_s


path = '/Users/ivanguseff/PycharmProjects/LitSim/results_normalised/N=4/'
auth1_auth1 = []
auth1_auth2 = []

for root, _, files in os.walk(path):
    for file in files:
        if file.startswith('.'):
            continue
        stats = txt_linesreader(root + '/' + file)[0]
        stats = eval(stats)
        if 'auth1–auth1' in root:
            book_values = list(stats.values())
            book_values = [list(i.values()) for i in book_values]
            auth1_auth1.extend(book_values)
        elif 'auth1–auth2' in root:
            book_values = list(stats.values())
            book_values = [list(i.values()) for i in book_values]
            auth1_auth2.extend(book_values)


accuracy = []
precision = []
recall = []
conf_matrix = []
for i in range(100):
    accuracy.append(training(auth1_auth1, auth1_auth2)[0])
    precision.append(training(auth1_auth1, auth1_auth2)[1])
    recall.append(training(auth1_auth1, auth1_auth2)[2])
    conf_matrix.append(training(auth1_auth1, auth1_auth2)[3])

print('Accuracy:', statistics.mean(accuracy))
print('Precision:', statistics.mean(precision))
print('Recall:', statistics.mean(recall))
