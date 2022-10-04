from typing import Dict, List


def precision_at_k(k: int, relevants: List[str], retrieved: List[str]) -> float:
    """
    Compute Precision at K
    """
    score = 0.0
    for p in retrieved[:k]:
        if p in relevants:
            score += 1.0
    return score / k


def recall_at_k(k: int, relevants: List[str], retrieved: List[str]) -> float:
    """
    Compute recall at K
    """
    score = 0.0
    for p in retrieved[:k]:
        if p in relevants:
            score += 1.0
    return score / len(relevants)


def average_precision(relevants: List[str], retrieved: List[str]) -> float:
    """
    Compute Average Precision
    """
    score = 0.0
    for i, p in enumerate(retrieved):

        k = i + 1
        if p in relevants:
            rel_k = 1
        else:
            rel_k = 0

        score += rel_k * precision_at_k(k, relevants, retrieved)

    return score / len(relevants)


def mean_average_precision(queries: List[Dict]) -> float:
    """
    Compute Mean Average Precision from a list of dict
    """
    if len(queries) == 0:
        raise ZeroDivisionError("Queries list can't be empty")

    score = 0.0
    for query in queries:

        score += average_precision(query["gold"], query["predicted"])

    return score / len(queries)
