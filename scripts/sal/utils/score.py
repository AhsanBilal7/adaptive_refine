"""Score aggregation utilities."""


def aggregate_scores(scores, strategy="last"):
    """
    Aggregate a list of per-step PRM scores into a single score.

    Args:
        scores: List of float scores (one per reasoning step).
        strategy: "last" (final step), "min" (worst step), "mean" (average).
    """
    if not scores:
        return 0.0
    if strategy == "last":
        return scores[-1]
    elif strategy == "min":
        return min(scores)
    elif strategy == "mean":
        return sum(scores) / len(scores)
    else:
        return scores[-1]


def score(dataset, config):
    """Placeholder for dataset-level scoring / evaluation."""
    pass
