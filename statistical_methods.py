from scipy import stats


"""
This module incorporates different functions with statistical methods.
"""


def tanimoto(a: dict, b: dict) -> float:
    """
    Measures Tanimoto coefficient for the dictionaries a and b with elements as keys and weights as values.
    """
    sum_min = sum(min(a.get(k, 0), b.get(k, 0)) for k in set(a.keys()).union(b.keys()))
    sum_max = sum(max(a.get(k, 0), b.get(k, 0)) for k in set(a.keys()).union(b.keys()))
    return sum_min / sum_max if sum_max != 0 else 0.0


def jaccard(a: set, b: set) -> float:
    """
    Measures Jaccard coefficient for the sets a and b.
    """
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)


def ttest_independent(list_1, list_2):
    t_stat, p_value = stats.ttest_ind(list_1, list_2, nan_policy='omit')
    return t_stat, p_value


def ttest_related(list_1, list_2):
    if len(list_1) >= len(list_2):
        bigger_list = list_1
        smaller_list = list_2
    elif len(list_1) < len(list_2):
        bigger_list = list_2
        smaller_list = list_1

    t_stat_rel, p_value_rel = stats.ttest_rel(bigger_list[:len(smaller_list)], smaller_list, nan_policy='omit')
    return t_stat_rel, p_value_rel

