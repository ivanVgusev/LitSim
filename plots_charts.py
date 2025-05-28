from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def dendrogram(values, names, title="Dendrogram"):
    Z = hierarchy.linkage(values, method='ward')
    plt.figure(figsize=(len(names), 5))
    hierarchy.dendrogram(Z, labels=np.array(names), show_leaf_counts=True)
    plt.title(title)
    plt.show()


def histogramm(list1, list2, title='Histogramm'):
    plt.figure(figsize=(13, 10), dpi=80)
    sns.distplot(list1, color="red", label="First element", hist_kws={'alpha': .7},
                 kde_kws={'linewidth': 3})
    sns.distplot(list2, color="dodgerblue", label="Second element", hist_kws={'alpha': .7},
                 kde_kws={'linewidth': 3})

    plt.title(title, fontsize=22)
    plt.legend()

    min_value = 0
    max_value = max(list1 + list2)
    step_size = 1
    plt.xticks(np.arange(min_value, max_value + step_size, step_size))
    plt.show()


def draw_scatter_plot(x_data: list, y_data: list, title="Scatter Plot"):
    if len(x_data) != len(y_data):
        raise ValueError("Длина списков должна совпадать")
    fig, ax = plt.subplots()

    ax.scatter(x_data, y_data)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

