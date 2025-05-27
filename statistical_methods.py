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
